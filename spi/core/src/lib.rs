wit_bindgen::generate!({
    world: "imports",
});
//
// struct Core;
//
// impl Guest for Core {
//     fn get_model_id() -> String {
//         "model_id".to_string()
//     }
// }

//
// struct KVStore;
//
// impl Guest for KVStore {
//     fn replace_value(key: String, value: String) -> Option<String> {
//         let kv = wasi_mindmap::kv_store::kvdb::Connection::new();
//         // replace
//         let old = kv.get(&key);
//         kv.set(&key, &value);
//         old
//     }
// }
//
// export!(KVStore);
