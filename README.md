Meemo Bot – Your Offline, Self-Learning Terminal AI Friend 🤖

Meemo Bot is a fully offline, terminal-based AI chat assistant designed for Termux on Android. Built with only Python standard libraries (no ML frameworks, no APIs), it learns from every conversation, adapts to your mood, and evolves over time to become your personalized companion.

---

✨ Key Features

· Self-Evolving Intelligence – Learns from interactions, adjusts response weights, and improves answer selection automatically.
· 100% Offline – No internet required. All knowledge stored locally in SQLite.
· Mood & Sentiment Tracking – Detects if you're happy, sad, or angry; adjusts tone accordingly.
· Personality Modes – Switch between casual (friendly) and formal (polite) styles.
· Manual Training – Add Q&A pairs directly.
· Bulk Knowledge Injection – Paste JSON/CSV data to instantly boost knowledge.
· Terminal-Based Menu – Easy navigation with options to chat, train, view, correct, reset, export, and cluster knowledge.
· Lightweight & Fast – Optimized for 2–4GB RAM Android devices; handles thousands of entries efficiently.

---

🧠 How It Works (No ML Libraries)

· Tokenization & stop-word removal
· Bag-of-words keyword extraction
· Cosine similarity matching
· Weighted response ranking
· Reinforcement-style weight updates (increase on use, decrease on wrong feedback)
· Sentiment analysis using predefined word lists
· Clustering to merge duplicate entries

---

📱 Installation on Termux

```bash
pkg update && pkg upgrade
pkg install python
git clone https://github.com/yourusername/meemo-bot.git
cd meemo-bot
python meemo_bot.py
```

Or simply copy the single Python file to your Termux home and run it.

---

🚀 Quick Start

1. Run python meemo_bot.py
2. Choose 1. Chat to start talking.
3. When Meemo doesn't know something, it asks you to teach it.
4. Use the menu to:
   · Correct last response if wrong
   · View knowledge base
   · Inject bulk knowledge via paste (JSON/CSV)
   · Toggle personality mode
   · Export database to CSV

---

📂 Project Structure (Single File)

Everything is contained in meemo_bot.py. When run, it creates:

· meemo_knowledge.db – SQLite database
· meemo_chat_log.txt – Conversation log
· meemo_export.csv – Exported knowledge (optional)

---

📦 Example Knowledge Injection (Paste Format)

JSON:

```json
[
  {"input": "Hello", "response": "Hi there!"},
  {"input": "How are you?", "response": "I'm fine, thanks!"}
]
```

CSV:

```csv
user_input,bot_response,weight
Hello,Hi there!,1.5
How are you?,I'm fine, thanks!
```

Just paste and press Ctrl+D – Meemo learns instantly!

---

🛠️ Built With

· Python 3
· SQLite3 (standard library)
· No external dependencies – truly portable!

---

🤝 Contributing

Feel free to fork, enhance, or adapt Meemo Bot. Ideas for improvement: better clustering, multi-language support, or a web UI – all while keeping it offline and dependency-free.

---

📄 License

MIT – use it, modify it, share it.

---

Meemo Bot – Your friend in the terminal. Forever offline, forever learning.
