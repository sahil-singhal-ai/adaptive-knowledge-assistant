
def trim_conversation_to_fit(conversation_turns, tokenizer, max_tokens_allowed):

    reversed_turns = list(reversed(conversation_turns))
    kept = []
    total_tokens = 0

    for turn in reversed_turns:
        turn_tokens = len(tokenizer(turn)["input_ids"])

        if total_tokens + turn_tokens > max_tokens_allowed:
            break

        kept.append(turn)
        total_tokens += turn_tokens

    kept.reverse()
    return kept
