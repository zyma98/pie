#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[inferlib_macros::wit_enum(interface = "queues")]
pub(crate) enum Priority {
    Low,
    Normal,
    High,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[inferlib_macros::wit_enum(interface = "queues")]
pub(crate) enum ResourceType {
    Adapter,
    KvPage,
    Embed,
}
