use serde::Serialize;
use serde_json::Value;

/// Represents a single tool call.
#[derive(Serialize, Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Represents a single message in the conversation history.
#[derive(Serialize, Clone, Debug)]
pub(crate) struct Message {
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
    pub fn new(template: String) -> Result<Self, String> {
        let mut env = minijinja::Environment::new();
        env.add_template_owned("chat", template)
            .map_err(|e| format!("Failed to compile chat template: {}", e))?;

        Ok(ChatFormatter {
            env,
            messages: Vec::new(),
        })
    }

    pub fn add_system<T: ToString>(&mut self, content: T) {
        self.messages.push(Message {
            role: "system".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: None,
        });
    }

    pub fn add_user<T: ToString>(&mut self, content: T) {
        self.messages.push(Message {
            role: "user".to_string(),
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: None,
        });
    }

    pub fn add_assistant<T: ToString>(&mut self, content: T) {
        self.add_assistant_response(content, None::<&str>, None);
    }

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
