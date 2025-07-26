# Chatbot Project

A conversational AI chatbot built using LangChain, OpenAI, and LangGraph. This project demonstrates a simple yet extensible framework for building chatbots with conversation history, state management, and colored terminal output.

## Features

- Conversational interface with persistent history
- Uses OpenAI's GPT-4o-mini model for responses
- State management via LangGraph
- Colored terminal output for improved readability
- Conversation history saved to a text file

## Requirements

- Python 3.8+
- OpenAI API key (set in `.env`)
- Dependencies listed in `pyproject.toml` and `uv.lock`

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nishant5790/chat-bot.git
   cd chat-bot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or use your preferred environment manager (e.g., `uv`, `poetry`).

3. **Set up environment variables:**
   - Create a `.env` file in the project root.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

## Usage

Run the chatbot from the terminal:

```bash
python main.py
```

- Enter your message when prompted.
- Type `exit` to end the conversation.
- The conversation history will be saved to `conversation_history.txt`.

## Project Structure

```
main.py                   # Main chatbot logic
conversation_history.txt  # Saved chat history
pyproject.toml            # Project dependencies
uv.lock                   # Dependency lock file
README.md                 # Project documentation
```

## How It Works

- The chatbot uses LangChain's message objects (`HumanMessage`, `AIMessage`) to track conversation.
- LangGraph manages the state and flow of the conversation.
- Each user input is processed, and the AI's response is appended to the conversation history.
- Colored output is provided for better readability in the terminal.
- At the end of the session, the full conversation is written to `conversation_history.txt`.

## Customization

- **Model:** Change the model in `main.py` by modifying `ChatOpenAI(model="gpt-4o-mini")`.
- **Output Colors:** Adjust colors in the `print_color` module.
- **Conversation Flow:** Extend the state graph for more complex dialog management.

## Dependencies

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [OpenAI](https://github.com/openai/openai-python)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- [print_color](custom module for colored output)

## License

This project is licensed under the MIT License.

## Author

Nishant Kumar
