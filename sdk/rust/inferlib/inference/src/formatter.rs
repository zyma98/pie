use crate::chat::{self, ChatFormatter as TemplateChatFormatter};

#[derive(Clone, Debug)]
#[inferlib_macros::wit_record]
pub(crate) struct ToolCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

pub(crate) struct ChatFormatter {
    formatter: TemplateChatFormatter,
}

#[inferlib_macros::shared_resource]
impl ChatFormatter {
    pub(crate) fn new(template: String) -> Self {
        let formatter = TemplateChatFormatter::new(template)
            .expect("Failed to create chat formatter: invalid template");
        ChatFormatter { formatter }
    }

    pub(crate) fn add_system(&mut self, content: String) {
        self.formatter.add_system(content);
    }

    pub(crate) fn add_user(&mut self, content: String) {
        self.formatter.add_user(content);
    }

    pub(crate) fn add_assistant(&mut self, content: String) {
        self.formatter.add_assistant(content);
    }

    pub(crate) fn add_assistant_response(
        &mut self,
        content: String,
        reasoning: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    ) {
        let internal_tool_calls = tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| {
                    let args: serde_json::Value = serde_json::from_str(&tc.arguments)
                        .unwrap_or(serde_json::Value::String(tc.arguments));
                    chat::MessageToolCall {
                        name: tc.name,
                        arguments: args,
                    }
                })
                .collect()
        });

        self.formatter
            .add_assistant_response(content, reasoning, internal_tool_calls);
    }

    pub(crate) fn add_tool(&mut self, content: String) {
        self.formatter.add_tool(content);
    }

    pub(crate) fn has_messages(&self) -> bool {
        self.formatter.has_messages()
    }

    pub(crate) fn clear(&mut self) {
        self.formatter.clear();
    }

    pub(crate) fn render(&self, add_generation_prompt: bool, begin_of_sequence: bool) -> String {
        self.formatter
            .render(add_generation_prompt, begin_of_sequence)
    }
}
