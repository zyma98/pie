class Engine:
    def __init__(self, command):
        self.command = command

        # llm
        # tokens

    def allocate_blocks(self):
        pass

    def deallocate_blocks(self):
        pass

    def allocate_embeds(self):
        pass

    def deallocate_embeds(self):
        pass

    def allocate_dists(self):
        ...

    def deallocate_dists(self):
        ...

    def embed_text(self):
        ...

    def embed_image(self):
        ...

    def fill_block(self):
        ...

    def mask_block(self):
        ...

    def copy_block(self):
        ...

    def decode_token_distribution(self):
        ...

    def sample_top_k_request(self):
        ...

    def get_token_distribution(self):
        ...
