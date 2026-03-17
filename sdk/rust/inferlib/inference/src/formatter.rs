use crate::chat::{self, ChatFormatter};
use crate::exports::inferlib::inference::formatter::{GuestChatFormatter, ToolCall as WitToolCall};

use std::cell::RefCell;

pub(crate) struct ChatFormatterImpl {
    formatter: RefCell<ChatFormatter>,
}

impl GuestChatFormatter for ChatFormatterImpl {
    fn new(template: String) -> Self {
        let formatter = ChatFormatter::new(template)
            .expect("Failed to create chat formatter: invalid template");
        ChatFormatterImpl {
            formatter: RefCell::new(formatter),
        }
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
        let internal_tool_calls = tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| {
                    let args: serde_json::Value = serde_json::from_str(&tc.arguments)
                        .unwrap_or(serde_json::Value::String(tc.arguments));
                    chat::ToolCall {
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
