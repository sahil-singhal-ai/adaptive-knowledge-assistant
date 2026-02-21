class ChatMemory:
    def __init__(self, max_messages=8):
        self.messages = []
        self.max_messages = max_messages

        # Default system instruction
        self.messages.append({
            "role": "system",
            "content": "You are a helpful document-based AI assistant."
        })

    def add_user(self, content):
        self.messages.append({
            "role": "user",
            "content": content
        })
        self._trim()

    def add_assistant(self, content):
        self.messages.append({
            "role": "assistant",
            "content": content
        })
        self._trim()

    def get_messages(self):
        return self.messages

    def get_recent_conversation(self,n_turns=2):
        return self.messages[-(n_turns * 2):]

    def _trim(self):
        # Keep system + last N messages
        system_message = self.messages[0]
        other_messages = self.messages[1:]

        if len(other_messages) > self.max_messages:
            other_messages = other_messages[-self.max_messages:]

        self.messages = [system_message] + other_messages

    def clear(self):
      #wipes out all memory
      system_message = self.messages[0]
      self.messages = [system_message]