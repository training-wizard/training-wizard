"""Dataset specifications."""

from logging import Logger

import structlog
from datasets import Dataset

logger: Logger = structlog.get_logger()


def validate_preference_dataset(ds: Dataset):
    """Validate that the dataset has the required columns for CPO training.

    Dataset Format:
    The dataset must follow one of these two formats:

    1. Standard Format:
    {
        "prompt": "What is the capital of France?",
        "chosen": "Paris is the capital of France.",
        "rejected": "I believe London is the capital of France."
    }

    2. Conversational Format (System Message is optional):
    {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "chosen": [
            {"role": "assistant", "content": "Paris is the capital of France."}
        ],
        "rejected": [
            {"role": "assistant", "content": "I believe London is the capital of France."}
        ]
    }
    """
    dataset_columns = set(ds.column_names)

    # Check for standard preference format
    required_columns = {"chosen", "rejected", "prompt"}

    missing_columns = required_columns - dataset_columns
    if missing_columns:
        raise ValueError(
            "Dataset must follow either standard or conversational preference format.\n"
            f"Standard format requires: {required_columns}\n"
            f"Missing columns: {missing_columns}"
        )

    # Validate first row format
    first_row = ds[0]
    for column in ["chosen", "rejected", "prompt"]:
        value = first_row[column]

        # Check if it's a string or list of chat messages
        is_string = isinstance(value, str)
        is_chat_list = isinstance(value, list) and all(
            isinstance(msg, dict) and "role" in msg and "content" in msg for msg in value
        )

        if not (is_string or is_chat_list):
            raise ValueError(
                f"Column '{column}' must contain either strings or lists of chat messages.\n"
                "Chat messages must be dictionaries with 'role' and 'content' keys.\n"
                f"Got type: {type(value)}"
            )


def validate_prompt_only_dataset(ds: Dataset):
    """Prompt-only dataset for Online DPO.

    Dataset Format:
    The dataset must follow one of these two formats:

    1. Standard Format:
    {
        "prompt": "What is the capital of France?"
    }

    2. Conversational Format (System Message is optional):
    {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    }
    """
    dataset_columns = set(ds.column_names)

    # Check for standard prompt-only format
    required_columns = {"prompt"}
    missing_columns = required_columns - dataset_columns
    if missing_columns:
        raise ValueError(
            "Dataset must follow either standard or conversational prompt-only format.\n"
            f"Standard format requires: {required_columns}\n"
            f"Found columns: {dataset_columns}\n\n"
            "This recipe only requires prompts - the responses will be generated during training."
        )

    # For conversational format, validate last message is from user
    first_row = ds[0]
    is_conversational = isinstance(first_row["prompt"], list)
    if is_conversational:
        messages = first_row["prompt"]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError(
                "In prompt-only conversational format, the last message must be from the user.\n"
                "This recipe generates assistant responses during training."
            )


def validate_sequence_classification_dataset(ds: Dataset):
    """Validate that the dataset has the required columns for classification training.

    Dataset Format:
    Single label:
    {
        "text": "What is the capital of France?",
        "labels": 0
    }

    Multi-label:
    {
        "text": "What is the capital of France?",
        "labels": [1, 1]
    }
    """
    dataset_columns = set(ds.column_names)
    required_columns = {"text", "labels"}
    missing_columns = required_columns - dataset_columns
    if "label" in dataset_columns and "labels" in missing_columns:
        raise ValueError("Found column 'label' instead of 'labels' for classification. Please rename to 'labels'.")

    if missing_columns:
        raise ValueError(f"Dataset must have columns: {required_columns} - Missing columns: {missing_columns}")

    # Validate that labels are integers
    first_row = ds[0]
    if isinstance(first_row["labels"], list):
        if not all(isinstance(x, int) for x in first_row["labels"]):
            raise ValueError("For multi-label classification, all labels must be integers")
    elif not isinstance(first_row["labels"], int):
        raise ValueError(f"'labels' column must contain integers, found {type(first_row['labels'])}")


def validate_regression_dataset(ds: Dataset):
    """Validate that the dataset has the required columns for regression training.

    Dataset Format:
    {
        "text": "What is the temperature today?",
        "labels": 72.5
    }
    """
    dataset_columns = set(ds.column_names)
    required_columns = {"text", "labels"}
    missing_columns = required_columns - dataset_columns
    if "label" in dataset_columns and "labels" in missing_columns:
        raise ValueError("Found column 'label' instead of 'labels' for regression. Please rename to 'labels'.")

    if missing_columns:
        raise ValueError(f"Dataset must have columns: {required_columns} - Missing columns: {missing_columns}")

    # Validate that labels are floats
    first_row = ds[0]
    if not isinstance(first_row["labels"], int | float | list):
        raise ValueError(f"'labels' column must contain ints, floats, or lists, found {type(first_row['labels'])}")


def validate_seq2seq_dataset(ds: Dataset):
    """Validate that the dataset has the required columns for seq2seq training.

    Dataset Format:
    {
        "source": "What is the capital of France?",
        "target": "Paris is the capital of France."
    }
    """
    dataset_columns = set(ds.column_names)
    required_columns = {"source", "target"}
    missing_columns = required_columns - dataset_columns
    if missing_columns:
        raise ValueError(f"Dataset must have columns: {required_columns} - Missing columns: {missing_columns}")
    # Validate that source and target are strings
    first_row = ds[0]
    if not isinstance(first_row["source"], str):
        raise ValueError(f"'source' column must contain strings, found {type(first_row['source'])}")
    if not isinstance(first_row["target"], str):
        raise ValueError(f"'target' column must contain strings, found {type(first_row['target'])}")


def validate_instruction_dataset(ds: Dataset):
    """Validate that the dataset has the required columns for instruction tuning.

    Dataset Format (System Message is optional):
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris is the capital of France."}
        ]
    }
    """
    dataset_columns = set(ds.column_names)
    required_columns = {"messages"}
    missing_columns = required_columns - dataset_columns
    if missing_columns:
        raise ValueError(f"Dataset must have columns: {required_columns} - Missing columns: {missing_columns}")

    # Validate first row
    first_row = ds[0]
    messages = first_row["messages"]

    # Check messages is a list
    if not isinstance(messages, list):
        raise ValueError(f"'messages' column must contain lists, found {type(messages)}")

    # Check each message is a dict with required keys
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(f"Each message must be a dictionary, found {type(msg)}")

        required_keys = {"role", "content"}
        missing_keys = required_keys - set(msg.keys())
        if missing_keys:
            raise ValueError(f"Each message must have keys: {required_keys} - Missing keys: {missing_keys}")

        # Validate types
        if not isinstance(msg["role"], str):
            raise ValueError(f"Message 'role' must be a string, found {type(msg['role'])}")
        if not isinstance(msg["content"], str):
            raise ValueError(f"Message 'content' must be a string, found {type(msg['content'])}")

    # Validate that the last message is an assistant message
    if not messages[-1]["role"] == "assistant":
        raise ValueError("The last message must be an assistant message")
