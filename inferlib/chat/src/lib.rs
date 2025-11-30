// Generate WIT bindings
wit_bindgen::generate!({
    path: "wit",
    world: "chat-formatter",
});

use exports::inferlib::chat::formatter::{Guest, GuestChatFormatter, ToolCall as WitToolCall};
use serde::Serialize;
use serde_json::Value;
use std::cell::RefCell;

/// Represents a single tool call.
#[derive(Serialize, Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Represents a single message in the conversation history.
/// This struct is no longer generic over a lifetime. All string data is owned.
#[derive(Serialize, Clone, Debug)]
struct Message {
    role: String,
    content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug)]
pub struct ChatFormatter {
    env: minijinja::Environment<'static>,
    messages: Vec<Message>,
}

impl ChatFormatter {
    /// Creates a new Conversation instance with a given chat template.
    /// The template is compiled once at construction time and owned by the Environment.
    /// Returns an error string if the template has syntax errors.
    pub fn new(template: String) -> Result<Self, String> {
        let mut env = minijinja::Environment::new();
        env.add_template_owned("chat", template)
            .map_err(|e| format!("Failed to compile chat template: {}", e))?;

        Ok(ChatFormatter {
            env,
            messages: Vec::new(),
        })
    }

    /// Adds a system message to the conversation.
    /// The `content` parameter now accepts any type that implements `ToString`.
    pub fn add_system<T: ToString>(&mut self, content: T) {
        self.messages.push(Message {
            role: "system".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: None,
        });
    }

    /// Adds a user message to the conversation.
    pub fn add_user<T: ToString>(&mut self, content: T) {
        self.messages.push(Message {
            role: "user".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: None,
        });
    }

    /// Adds a simple assistant message with only text content.
    pub fn add_assistant<T: ToString>(&mut self, content: T) {
        self.add_assistant_response(content, None::<&str>, None);
    }

    /// Adds a comprehensive assistant response, optionally including reasoning and tool calls.
    /// Both `content` and `reasoning` are now generic over `ToString`.
    pub fn add_assistant_response<T: ToString, R: ToString>(
        &mut self,
        content: T,
        reasoning: Option<R>,
        tool_calls: Option<Vec<ToolCall>>,
    ) {
        self.messages.push(Message {
            role: "assistant".to_string(),
            content: content.to_string(),
            reasoning_content: reasoning.map(|s| s.to_string()),
            tool_calls,
        });
    }

    /// Adds a tool response message to the conversation.
    pub fn add_tool<T: ToString>(&mut self, content: T) {
        self.messages.push(Message {
            role: "tool".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: None,
        });
    }

    pub fn has_messages(&self) -> bool {
        !self.messages.is_empty()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Renders the entire conversation into a single formatted string prompt.
    /// Uses the pre-compiled template stored in the Environment for maximum efficiency.
    pub fn render(&self, add_generation_prompt: bool, begin_of_sequence: bool) -> String {
        let tmpl = self
            .env
            .get_template("chat")
            .expect("Template should exist in environment");

        tmpl.render(minijinja::context! {
            messages => &self.messages,
            add_generation_prompt,
            begin_of_sequence,
        })
        .expect("Failed to render template")
    }
}

struct FormatterImpl;

impl Guest for FormatterImpl {
    type ChatFormatter = ChatFormatterImpl;
}

struct ChatFormatterImpl {
    formatter: RefCell<ChatFormatter>,
}

impl GuestChatFormatter for ChatFormatterImpl {
    fn new(template: String) -> Result<Self, String> {
        let formatter = ChatFormatter::new(template)?;
        Ok(ChatFormatterImpl {
            formatter: RefCell::new(formatter),
        })
    }

    fn add_system(&self, content: String) {
        self.formatter.borrow_mut().add_system(content);
    }

    fn add_user(&self, content: String) {
        self.formatter.borrow_mut().add_user(content);
    }

    fn add_assistant(&self, content: String) {
        self.formatter.borrow_mut().add_assistant(content);
    }

    fn add_assistant_response(
        &self,
        content: String,
        reasoning: Option<String>,
        tool_calls: Option<Vec<WitToolCall>>,
    ) {
        // Convert WIT tool calls to internal tool calls
        let internal_tool_calls = tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| {
                    let args: Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(Value::String(tc.arguments));
                    ToolCall {
                        name: tc.name,
                        arguments: args,
                    }
                })
                .collect()
        });

        self.formatter
            .borrow_mut()
            .add_assistant_response(content, reasoning, internal_tool_calls);
    }

    fn add_tool(&self, content: String) {
        self.formatter.borrow_mut().add_tool(content);
    }

    fn has_messages(&self) -> bool {
        self.formatter.borrow().has_messages()
    }

    fn clear(&self) {
        self.formatter.borrow_mut().clear();
    }

    fn render(&self, add_generation_prompt: bool, begin_of_sequence: bool) -> String {
        self.formatter
            .borrow()
            .render(add_generation_prompt, begin_of_sequence)
    }
}

export!(FormatterImpl);
