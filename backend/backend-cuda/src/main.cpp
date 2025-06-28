#include <iostream>
#include <string>
#include <optional>        // For std::optional
#include <map>             // For std::map
#include <vector>          // For std::vector
#include <filesystem>      // For std::filesystem
#include <fstream>         // For std::ifstream
#include <stdexcept>       // For std::runtime_error
#include <cstdlib>         // For std::getenv
#include <cstring>         // For std::strlen


#include "utils.hpp"

#include "handshake.pb.h"
#include "l4m.pb.h"

#include <zmq.hpp>
#include <CLI/CLI.hpp>        
#include <toml++/toml.hpp>       

// Holds the final, merged configuration for the application.
struct AppConfig
{
    // Network
    std::string host = "localhost";
    int port = 10123;
    std::string controller_host = "localhost";
    int controller_port = 9123;
    std::optional<std::string> auth_token;

    // Model & Cache
    std::optional<std::string> model_name;
    std::string version = "";
    std::filesystem::path cache_dir;

    // Engine Parameters
    int kv_page_size = 32;
    int dist_size = 32;
    int max_num_kv_pages = 1000;
    int max_num_embeds = 50000;

    // Hardware
    std::string device = "cuda:0";
    std::string dtype = "bfloat16";
};

// Corresponds to the [architecture] table in the model's TOML file.
struct L4maConfig
{
    std::string type;
    int num_layers;
    int num_query_heads;
    int num_key_value_heads;
    int head_size;
    int hidden_size;
    int intermediate_size;
    int vocab_size;
    bool use_qkv_bias;
    float rms_norm_eps;
    float rope_factor;
    float rope_high_frequency_factor;
    float rope_low_frequency_factor;
    float rope_theta;
    std::string device = "cuda:0";
    // torch::dtype dtype = torch::kBFloat16; // Placeholder for a specific dtype
};

// Corresponds to the [tokenizer] table in the model's TOML file.
struct TokenizerInfo
{
    std::string type;
    std::map<int, std::vector<uint8_t>> merge_table;
    std::string split_regex;
    std::map<int, std::string> special_tokens;
};

// Top-level structure holding the entire parsed model configuration.
struct ModelMetadata
{
    std::string name;
    std::string description;
    std::string version;
    std::vector<std::string> parameters;
    L4maConfig architecture;
    TokenizerInfo tokenizer;
    std::string template_type;
    std::string template_content;
};


// ======================================================================================
// MARK: - Helper and Parsing Functions
// ======================================================================================

// Loads merge rules from a Llama 3 vocabulary file.
std::map<int, std::vector<uint8_t>> load_merge_rules(const std::filesystem::path& path) {
    std::map<int, std::vector<uint8_t>> merge_rules;
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to load vocabulary file: " + path.string());
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
            continue; // Skip empty or whitespace-only lines
        }

        std::string::size_type pos = line.find(' ');
        if (pos == std::string::npos) {
             throw std::runtime_error("Error on line " + std::to_string(line_number) + ": expected 2 parts, but found 1.");
        }
        
        std::string b64_token = line.substr(0, pos);
        std::string rank_str = line.substr(pos + 1);

        try {
            int rank = std::stoi(rank_str);
            merge_rules[rank] = utils::base64_decode(b64_token);
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error("Error on line " + std::to_string(line_number) + ": failed to parse rank '" + rank_str + "' as an integer.");
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("Error on line " + std::to_string(line_number) + ": rank '" + rank_str + "' is out of range.");
        }
    }
    return merge_rules;
}

// Parses the model's metadata TOML file with detailed error checking.
ModelMetadata parse_model_metadata(const std::filesystem::path& metadata_path)
{
    if (!std::filesystem::exists(metadata_path)) {
        throw std::runtime_error("Metadata file not found at: " + metadata_path.string());
    }
    
    const auto metadata_dir = metadata_path.parent_path();

    toml::table tbl;
    try {
        tbl = toml::parse_file(metadata_path.string());
    } catch (const toml::parse_error& err) {
        throw std::runtime_error("Failed to parse model metadata TOML '" + metadata_path.string() + "': " + std::string(err.what()));
    }

    ModelMetadata metadata;

    // Helper to get a required value from a TOML table.
    auto get_required = [&](const toml::table& t, const std::string& key, const std::string& section) {
        auto node = t[key];
        if (!node) {
            throw std::runtime_error("Missing required key '" + key + "' in section '[" + section + "]' of TOML file: " + metadata_path.string());
        }
        return node;
    };
    
    // --- Parse top-level keys ---
    metadata.name = get_required(tbl, "name", "top-level").value_or("");
    metadata.description = get_required(tbl, "description", "top-level").value_or("");
    metadata.version = get_required(tbl, "version", "top-level").value_or("");

    if (auto params_node = get_required(tbl, "parameters", "top-level").as_array()) {
        for (const auto& elem : *params_node) {
            metadata.parameters.push_back(elem.value_or(""));
        }
    }

    // --- Parse Architecture ---
    auto arch_data = get_required(tbl, "architecture", "top-level").as_table();
    auto rope_data = get_required(*arch_data, "rope", "architecture").as_table();
    
    metadata.architecture = {
        .type = get_required(*arch_data, "type", "architecture").value_or(""),
        .num_layers = get_required(*arch_data, "num_layers", "architecture").value_or(0),
        .num_query_heads = get_required(*arch_data, "num_query_heads", "architecture").value_or(0),
        .num_key_value_heads = get_required(*arch_data, "num_key_value_heads", "architecture").value_or(0),
        .head_size = get_required(*arch_data, "head_size", "architecture").value_or(0),
        .hidden_size = get_required(*arch_data, "hidden_size", "architecture").value_or(0),
        .intermediate_size = get_required(*arch_data, "intermediate_size", "architecture").value_or(0),
        .vocab_size = get_required(*arch_data, "vocab_size", "architecture").value_or(0),
        .use_qkv_bias = false, // Hardcoded
        .rms_norm_eps = 1e-05f, // Hardcoded
        .rope_factor = static_cast<float>(get_required(*rope_data, "factor", "architecture.rope").value_or(0.0)),
        .rope_high_frequency_factor = static_cast<float>(get_required(*rope_data, "high_frequency_factor", "architecture.rope").value_or(0.0)),
        .rope_low_frequency_factor = static_cast<float>(get_required(*rope_data, "low_frequency_factor", "architecture.rope").value_or(0.0)),
        .rope_theta = static_cast<float>(get_required(*rope_data, "theta", "architecture.rope").value_or(0.0)),
    };

    // --- Parse Tokenizer ---
    auto tokenizer_data = get_required(tbl, "tokenizer", "top-level").as_table();
    std::string vocab_file = get_required(*tokenizer_data, "vocabulary_file", "tokenizer").value_or("");
    auto vocab_full_path = metadata_dir / vocab_file;
    
    metadata.tokenizer.type = get_required(*tokenizer_data, "type", "tokenizer").value_or("");
    metadata.tokenizer.split_regex = get_required(*tokenizer_data, "split_regex", "tokenizer").value_or("");
    metadata.tokenizer.merge_table = load_merge_rules(vocab_full_path);

    if (auto special_tokens_node = get_required(*tokenizer_data, "special_tokens", "tokenizer").as_table()) {
        for (const auto& [k, v] : *special_tokens_node) {
            metadata.tokenizer.special_tokens[std::stoi(std::string(k))] = v.value_or("");
        }
    }

    // --- Parse Template ---
    auto template_data = get_required(tbl, "template", "top-level").as_table();
    metadata.template_type = get_required(*template_data, "type", "template").value_or("");
    metadata.template_content = get_required(*template_data, "content", "template").value_or("");

    return metadata;
}

// Determines the cache directory based on a clear priority order.
std::filesystem::path get_cache_dir(
    const std::optional<std::string>& cli_path,
    const toml::table& config_table)
{
    // 1. Command-line argument
    if (cli_path && !cli_path->empty()) {
        return *cli_path;
    }
    // 2. Value from TOML config file
    if (auto node = config_table["cache_dir"]; node.is_string()) {
        return node.as_string()->get();
    }
    // 3. PIE_HOME environment variable
    if (const char* env_pie_home = std::getenv("PIE_HOME")) {
        if (std::strlen(env_pie_home) > 0) {
            return env_pie_home;
        }
    }
    // 4. Default to user cache directory
    return utils::get_user_cache_dir() / "pie";
}

// Prints the final configuration that the application will run with.
void print_config(const AppConfig& config)
{
    std::cout << "--- Configuration ---\n"
              << "host: " << config.host << "\n"
              << "port: " << config.port << "\n"
              << "controller_host: " << config.controller_host << "\n"
              << "controller_port: " << config.controller_port << "\n"
              << "auth_token: " << config.auth_token.value_or("Not Set") << "\n"
              << "model: " << config.model_name.value_or("Not Set") << "\n"
              << "version: " << config.version << "\n"
              << "cache_dir: " << config.cache_dir << "\n"
              << "kv_page_size: " << config.kv_page_size << "\n"
              << "dist_size: " << config.dist_size << "\n"
              << "max_num_kv_pages: " << config.max_num_kv_pages << "\n"
              << "max_num_embeds: " << config.max_num_embeds << "\n"
              << "device: " << config.device << "\n"
              << "dtype: " << config.dtype << "\n"
              << "----------------------\n" << std::endl;
}

// ======================================================================================
// MARK: - Core Service Logic
// ======================================================================================

// Loads the model and its weights from the specified files.
// NOTE: `l4ma::L4maForCausalLM` and `ztensor::Reader` are placeholders for the actual
// model and tensor-loading libraries.
void load_model(const AppConfig& config, ModelMetadata& out_metadata)
{
    if (!config.model_name) {
        throw std::runtime_error("Model name must be specified via --model or config file.");
    }

    const auto model_dir = config.cache_dir / *config.model_name;
    const std::string metadata_filename = *config.model_name + (config.version.empty() ? "" : "-") + config.version + ".toml";
    const auto metadata_path = model_dir / metadata_filename;

    std::cout << "Loading model '" << *config.model_name << "' from: " << model_dir << std::endl;
    std::cout << "Reading metadata from: " << metadata_path << std::endl;

    out_metadata = parse_model_metadata(metadata_path);
    
   
    // TODO: Check for missing keys and print warnings
    // ...

    // TODO: Set model to evaluation mode
    // model->eval();

}


// Registers this service with the central controller.
// NOTE: This is a placeholder. A real implementation would require a WebSocket
// and MessagePack library.
void register_with_controller(const AppConfig& config, const std::string& service_endpoint)
{
    std::cout << "[Registration Thread] Attempting to register with controller at "
              << config.controller_host << ":" << config.controller_port << std::endl;
    
    // PSEUDOCODE for registration logic
}


int main(int argc, char* argv[])
{
    // --- 1. Argument Parsing Setup ---
    CLI::App app{"PIE C++ Backend Service"};
    app.allow_config_extras(true);

    AppConfig cli_config;
    std::optional<std::string> config_filepath_opt;
    std::optional<std::string> cache_dir_opt;
    
    app.add_option("--config", config_filepath_opt, "Path to a TOML configuration file.")->check(CLI::ExistingFile);
    app.add_option("--host", cli_config.host, "The hostname to bind to.");
    app.add_option("--port", cli_config.port, "The port number to listen on.");
    app.add_option("--controller-host", cli_config.controller_host, "The controller hostname.");
    app.add_option("--controller-port", cli_config.controller_port, "The controller port number.");
    app.add_option("--auth-token", cli_config.auth_token, "The authentication token for the controller.");
    app.add_option("--model", cli_config.model_name, "The model name (e.g., 'llama-3.2-1b-instruct').");
    app.add_option("--version", cli_config.version, "The version of the model.");
    app.add_option("--cache-dir", cache_dir_opt, "The directory for caching models.");
    app.add_option("--kv-page-size", cli_config.kv_page_size, "The KV page size.");
    app.add_option("--dist-size", cli_config.dist_size, "The distribution size.");
    app.add_option("--max-num-kv-pages", cli_config.max_num_kv_pages, "The maximum number of KV pages.");
    app.add_option("--max-num-embeds", cli_config.max_num_embeds, "The maximum number of embeddings.");
    app.add_option("--device", cli_config.device, "The device to use (e.g., 'cuda:0').");
    app.add_option("--dtype", cli_config.dtype, "The data type (e.g., 'bfloat16').");

    CLI11_PARSE(app, argc, argv);

    try
    {
        // --- 2. Configuration Loading and Merging ---
        toml::table config_from_file;
        if (config_filepath_opt) {
            try {
                config_from_file = toml::parse_file(*config_filepath_opt);
            } catch (const toml::parse_error& err) {
                std::cerr << "Error decoding TOML file '" << *config_filepath_opt << "': " << err.what() << std::endl;
                return 1;
            }
        }

        AppConfig final_config;
        final_config.host = app.count("--host") > 0 ? cli_config.host : config_from_file["host"].value_or(final_config.host);
        final_config.port = app.count("--port") > 0 ? cli_config.port : config_from_file["port"].value_or(final_config.port);
        final_config.controller_host = app.count("--controller-host") > 0 ? cli_config.controller_host : config_from_file["controller_host"].value_or(final_config.controller_host);
        final_config.controller_port = app.count("--controller-port") > 0 ? cli_config.controller_port : config_from_file["controller_port"].value_or(final_config.controller_port);
        final_config.model_name = app.count("--model") > 0 ? cli_config.model_name : config_from_file["model"].value<std::string>();

        if (app.count("--auth-token") > 0) {
            final_config.auth_token = cli_config.auth_token;
        } else if (auto token_node = config_from_file["auth_token"]; token_node.is_string()) {
            final_config.auth_token = token_node.as_string()->get();
        }

        final_config.model_name = app.count("--model") > 0 ? cli_config.model_name : config_from_file["model"].value<std::string>();
        final_config.version = app.count("--version") > 0 ? cli_config.version : config_from_file["version"].value_or(final_config.version);
        final_config.kv_page_size = app.count("--kv-page-size") > 0 ? cli_config.kv_page_size : config_from_file["kv_page_size"].value_or(final_config.kv_page_size);
        final_config.dist_size = app.count("--dist-size") > 0 ? cli_config.dist_size : config_from_file["dist_size"].value_or(final_config.dist_size);
        final_config.max_num_kv_pages = app.count("--max-num-kv-pages") > 0 ? cli_config.max_num_kv_pages : config_from_file["max_num_kv_pages"].value_or(final_config.max_num_kv_pages);
        final_config.max_num_embeds = app.count("--max-num-embeds") > 0 ? cli_config.max_num_embeds : config_from_file["max_num_embeds"].value_or(final_config.max_num_embeds);
        final_config.device = app.count("--device") > 0 ? cli_config.device : config_from_file["device"].value_or(final_config.device);
        final_config.dtype = app.count("--dtype") > 0 ? cli_config.dtype : config_from_file["dtype"].value_or(final_config.dtype);        
        final_config.cache_dir = get_cache_dir(cache_dir_opt, config_from_file);

        print_config(final_config);

        // --- 3. Load Model ---
        ModelMetadata model_metadata;
        load_model(final_config, model_metadata);
        std::cout << "Model loading complete." << std::endl;

        
    }
    catch (const std::exception& e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Server shutdown complete." << std::endl;
    return 0;
}
