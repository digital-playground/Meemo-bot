#!/usr/bin/env python3
"""
Meemo Bot - Single file, offline, self-learning chat assistant for Termux.
Uses only Python standard libraries and SQLite.
"""

import sqlite3
import os
import sys
import time
import datetime
import math
import string
import json
import csv
from collections import deque

# =============================================================================
# Database Manager
# =============================================================================
class Database:
    """Handles all SQLite operations for storing knowledge."""
    def __init__(self, db_path="meemo_knowledge.db"):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_table()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def _create_table(self):
        """Create knowledge table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT NOT NULL,
                keywords TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                usage_count INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_entry(self, user_input, keywords, response, weight=1.0):
        """Insert a new Q&A pair."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO knowledge (user_input, keywords, bot_response, weight)
            VALUES (?, ?, ?, ?)
        """, (user_input, keywords, response, weight))
        self.conn.commit()
        return cursor.lastrowid

    def add_entries_bulk(self, entries):
        """
        Insert multiple entries at once.
        entries: list of tuples (user_input, keywords, bot_response, weight)
        """
        cursor = self.conn.cursor()
        cursor.executemany("""
            INSERT INTO knowledge (user_input, keywords, bot_response, weight)
            VALUES (?, ?, ?, ?)
        """, entries)
        self.conn.commit()
        return cursor.rowcount

    def update_entry(self, entry_id, response=None, weight=None, usage_count=None):
        """Update an existing entry."""
        cursor = self.conn.cursor()
        fields = []
        values = []
        if response is not None:
            fields.append("bot_response = ?")
            values.append(response)
        if weight is not None:
            fields.append("weight = ?")
            values.append(weight)
        if usage_count is not None:
            fields.append("usage_count = ?")
            values.append(usage_count)
        if not fields:
            return
        query = f"UPDATE knowledge SET {', '.join(fields)} WHERE id = ?"
        values.append(entry_id)
        cursor.execute(query, values)
        self.conn.commit()

    def delete_all(self):
        """Reset the database (delete all rows)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM knowledge")
        self.conn.commit()

    def get_all_entries(self):
        """Return all entries as a list of dictionaries."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, user_input, keywords, bot_response, weight, usage_count FROM knowledge")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_entry_by_id(self, entry_id):
        """Fetch a single entry by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, user_input, keywords, bot_response, weight, usage_count FROM knowledge WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def export_to_csv(self, filename="meemo_export.csv"):
        """Export all data to CSV file."""
        entries = self.get_all_entries()
        if not entries:
            print("No data to export.")
            return
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'user_input', 'keywords', 'bot_response', 'weight', 'usage_count', 'timestamp'])
            writer.writeheader()
            for e in entries:
                # Fetch timestamp separately because get_all_entries doesn't include it
                cursor = self.conn.cursor()
                cursor.execute("SELECT timestamp FROM knowledge WHERE id = ?", (e['id'],))
                ts = cursor.fetchone()['timestamp']
                e['timestamp'] = ts
                writer.writerow(e)
        print(f"Exported to {filename}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

# =============================================================================
# Text Processor (Tokenization, Stopwords, Similarity, Sentiment)
# =============================================================================
class TextProcessor:
    """Handles all text processing: tokenization, stopwords, similarity, sentiment."""

    # Basic English stopwords
    STOPWORDS = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    ])

    # Sentiment word lists
    POSITIVE_WORDS = set(['good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful',
                          'fantastic', 'nice', 'best', 'super', 'perfect', 'glad', 'pleased'])
    NEGATIVE_WORDS = set(['bad', 'terrible', 'awful', 'horrible', 'sad', 'hate', 'worse',
                          'worst', 'upset', 'disappointed', 'angry', 'mad', 'furious'])
    ANGRY_WORDS = set(['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'])

    def __init__(self):
        self.punctuation = string.punctuation

    def tokenize(self, text):
        """Convert text to lowercase, remove punctuation, split into tokens."""
        text = text.lower()
        # Replace punctuation with spaces
        for p in self.punctuation:
            text = text.replace(p, ' ')
        return text.split()

    def remove_stopwords(self, tokens):
        """Filter out stopwords."""
        return [t for t in tokens if t not in self.STOPWORDS]

    def process(self, text):
        """Full pipeline: tokenize + remove stopwords. Returns list of keywords."""
        tokens = self.tokenize(text)
        return self.remove_stopwords(tokens)

    def cosine_similarity(self, set_a, set_b):
        """Compute cosine similarity between two sets of words (binary vectors)."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        denominator = math.sqrt(len(set_a) * len(set_b))
        return intersection / denominator if denominator else 0.0

    def sentiment_score(self, text):
        """Return a sentiment score: positive - negative normalized by word count."""
        tokens = self.tokenize(text)  # use all tokens, not just keywords
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in self.POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in self.NEGATIVE_WORDS)
        # Also check for angry words (extra negative)
        angry = sum(1 for t in tokens if t in self.ANGRY_WORDS)
        total = len(tokens)
        # Score between -1 and 1
        score = (pos - neg - angry) / total
        return score

# =============================================================================
# AI Engine (Learning, Response Selection, Weight Adjustment)
# =============================================================================
class AIEngine:
    """Core AI logic: similarity search, response ranking, reinforcement learning."""

    def __init__(self, db, processor):
        self.db = db
        self.processor = processor
        self.last_response_id = None  # For correction feedback

    def get_response(self, user_input):
        """
        Find best response for user input.
        Returns (response_text, confidence) or (None, None) if no good match.
        """
        keywords = self.processor.process(user_input)
        if not keywords:
            return None, None  # Empty input

        input_set = set(keywords)
        entries = self.db.get_all_entries()
        best_entry = None
        best_conf = -1.0

        for e in entries:
            # Convert stored keywords string to set
            if not e['keywords']:
                continue
            entry_set = set(e['keywords'].split())
            sim = self.processor.cosine_similarity(input_set, entry_set)
            # Confidence = similarity * weight (usage count could also be factored)
            conf = sim * e['weight']
            if conf > best_conf:
                best_conf = conf
                best_entry = e

        threshold = 0.2  # Minimum confidence to consider a match
        if best_conf < threshold or best_entry is None:
            return None, None

        # Update usage and weight for the selected response
        new_usage = best_entry['usage_count'] + 1
        # Small reward each time it's used
        new_weight = best_entry['weight'] + 0.01
        self.db.update_entry(best_entry['id'], weight=new_weight, usage_count=new_usage)

        self.last_response_id = best_entry['id']
        return best_entry['bot_response'], best_conf

    def wrong_feedback(self):
        """User said last response was wrong: decrease its weight."""
        if self.last_response_id is None:
            print("No last response to correct.")
            return False
        entry = self.db.get_entry_by_id(self.last_response_id)
        if not entry:
            print("Last response no longer exists.")
            return False
        new_weight = entry['weight'] * 0.9  # Reduce by 10%
        self.db.update_entry(self.last_response_id, weight=new_weight)
        print("Weight decreased for that response.")
        return True

    def correct_last_response(self, correct_text):
        """User provides correct answer for the last query: replace and increase weight."""
        if self.last_response_id is None:
            print("No last response to correct.")
            return False
        entry = self.db.get_entry_by_id(self.last_response_id)
        if not entry:
            print("Last response no longer exists.")
            return False
        new_weight = entry['weight'] + 0.5  # Increase significantly
        self.db.update_entry(self.last_response_id, response=correct_text, weight=new_weight)
        print("Last response corrected and weight increased.")
        return True

    def add_manual_entry(self, user_input, response, weight=1.0):
        """Manually add a Q&A pair."""
        keywords = self.processor.process(user_input)
        keywords_str = ' '.join(keywords)
        # Optional: check for near-duplicates to avoid clutter
        entries = self.db.get_all_entries()
        input_set = set(keywords)
        for e in entries:
            entry_set = set(e['keywords'].split())
            sim = self.processor.cosine_similarity(input_set, entry_set)
            if sim > 0.95:
                print(f"Warning: Very similar question exists (ID {e['id']}: '{e['user_input']}').")
                choice = input("Add anyway? (y/n): ").lower()
                if choice != 'y':
                    print("Entry not added.")
                    return None
                break
        entry_id = self.db.add_entry(user_input, keywords_str, response, weight)
        print(f"Added new entry with ID {entry_id}.")
        return entry_id

    def inject_knowledge_from_file(self, filepath):
        """
        Bulk inject knowledge from a JSON or CSV file.
        Expected JSON format: list of objects with keys: "input", "response", (optional "weight")
        Example: [{"input": "Hello", "response": "Hi there!", "weight": 1.5}, ...]
        CSV format: columns user_input, bot_response, weight (weight optional)
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        ext = os.path.splitext(filepath)[1].lower()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        self._process_injection_content(content, source_type=ext)

    def inject_knowledge_paste(self):
        """
        Read multi-line pasted content from stdin (until EOF/Ctrl+D) and inject.
        """
        print("\n--- Paste your knowledge data (JSON or CSV) ---")
        print("Instructions: Paste the content, then press Ctrl+D (or Ctrl+Z on Windows) on a new line to finish.")
        print("For JSON: provide a list of objects with 'input' and 'response' keys.")
        print("For CSV: provide header row 'user_input,bot_response,weight' and data rows.")
        print("------------------------------------------------")
        try:
            # Read all lines until EOF
            content = sys.stdin.read()
        except KeyboardInterrupt:
            print("\nInjection cancelled.")
            return
        if not content.strip():
            print("No input provided.")
            return
        self._process_injection_content(content, source_type="paste")

    def _process_injection_content(self, content, source_type="unknown"):
        """
        Common processing for injected content (either from file or paste).
        Tries to parse as JSON first; if fails, tries as CSV.
        """
        content = content.strip()
        if not content:
            print("Empty content.")
            return

        entries = []
        # Try JSON first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        print("Skipping non-dictionary item in JSON list.")
                        continue
                    if 'input' not in item or 'response' not in item:
                        print("Skipping item missing 'input' or 'response'.")
                        continue
                    user_input = item['input'].strip()
                    response = item['response'].strip()
                    weight = item.get('weight', 1.0)
                    if not user_input or not response:
                        print("Skipping empty input or response.")
                        continue
                    keywords = self.processor.process(user_input)
                    keywords_str = ' '.join(keywords)
                    entries.append((user_input, keywords_str, response, weight))
                print(f"Parsed as JSON: {len(entries)} valid entries.")
            else:
                print("JSON root is not a list. Treating as CSV...")
                # Not a list, fall through to CSV handling
                pass
        except json.JSONDecodeError:
            # Not JSON, try CSV
            pass

        # If no entries yet, try CSV
        if not entries:
            try:
                # Use StringIO to simulate file
                import io
                f = io.StringIO(content)
                reader = csv.reader(f)
                headers = next(reader)  # assume first row is header
                if len(headers) < 2:
                    print("CSV must have at least 'user_input' and 'bot_response' columns.")
                    return
                for row in reader:
                    if len(row) < 2:
                        continue
                    user_input = row[0].strip()
                    response = row[1].strip()
                    weight = float(row[2]) if len(row) > 2 and row[2].strip() else 1.0
                    if not user_input or not response:
                        continue
                    keywords = self.processor.process(user_input)
                    keywords_str = ' '.join(keywords)
                    entries.append((user_input, keywords_str, response, weight))
                print(f"Parsed as CSV: {len(entries)} valid entries.")
            except Exception as e:
                print(f"Failed to parse as CSV: {e}")
                return

        if not entries:
            print("No valid entries found in the provided data.")
            return

        # Bulk insert
        count = self.db.add_entries_bulk(entries)
        print(f"Injected {count} new knowledge entries.")

    def cluster_knowledge(self):
        """
        Simple clustering: merge entries with identical keyword sets.
        Keeps the one with higher weight, combines usage counts.
        """
        entries = self.db.get_all_entries()
        # Group by keyword string (exact match)
        groups = {}
        for e in entries:
            key = e['keywords']
            if key not in groups:
                groups[key] = []
            groups[key].append(e)

        merged = 0
        for key, group in groups.items():
            if len(group) <= 1:
                continue
            # Sort by weight descending, then usage
            group.sort(key=lambda x: (x['weight'], x['usage_count']), reverse=True)
            keep = group[0]
            to_delete = group[1:]
            total_usage = keep['usage_count'] + sum(e['usage_count'] for e in to_delete)
            # Update the kept entry
            self.db.update_entry(keep['id'], usage_count=total_usage)
            # Delete others
            cursor = self.db.conn.cursor()
            ids = [e['id'] for e in to_delete]
            cursor.execute(f"DELETE FROM knowledge WHERE id IN ({','.join('?'*len(ids))})", ids)
            self.db.conn.commit()
            merged += len(to_delete)
            print(f"Merged {len(to_delete)} entries into ID {keep['id']}.")

        if merged:
            print(f"Clustering complete. Merged {merged} entries.")
        else:
            print("No duplicate keyword sets found.")

# =============================================================================
# Chat Session (Handles Interaction, Mood, Personality, Logging)
# =============================================================================
class ChatSession:
    """Manages a chat session with mood tracking and personality adaptation."""

    def __init__(self, ai_engine, db, processor):
        self.ai = ai_engine
        self.db = db
        self.processor = processor
        self.mood_history = deque(maxlen=10)  # Last 10 sentiment scores
        self.personality = "casual"  # or "formal"
        self.log_file = "meemo_chat_log.txt"

    def _log(self, user_msg, bot_msg):
        """Append interaction to log file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] User: {user_msg}\n")
            f.write(f"[{timestamp}] Meemo: {bot_msg}\n\n")

    def _get_tone_prefix(self, sentiment):
        """Generate a prefix based on current mood and personality."""
        if self.mood_history:
            avg_mood = sum(self.mood_history) / len(self.mood_history)
        else:
            avg_mood = sentiment

        prefix = ""
        if avg_mood < -0.3:
            prefix = "I sense you're upset. "
        elif avg_mood > 0.3:
            prefix = "I'm glad you're feeling good! "

        if self.personality == "casual":
            # Make prefix more casual
            prefix = prefix.lower().replace("i ", "i, ")  # silly example
            if not prefix:
                prefix = "hey, "
        else:
            # Formal: ensure capitalization
            prefix = prefix.capitalize()

        return prefix

    def chat(self):
        """Start the chat loop."""
        print("\n--- Chat Mode (type 'exit' to return to menu) ---")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'exit':
                break
            if not user_input:
                continue

            # Sentiment analysis
            sentiment = self.processor.sentiment_score(user_input)
            self.mood_history.append(sentiment)

            # Get AI response
            response, conf = self.ai.get_response(user_input)

            if response is None:
                print("Meemo: I don't know how to respond. Please teach me.")
                teach = input("What should I have said? ").strip()
                if teach:
                    self.ai.add_manual_entry(user_input, teach)
                    bot_msg = teach
                    print("Meemo: Thank you, I learned that.")
                else:
                    bot_msg = "Okay, maybe next time."
                    print(f"Meemo: {bot_msg}")
            else:
                prefix = self._get_tone_prefix(sentiment)
                bot_msg = prefix + response
                print(f"Meemo: {bot_msg} (confidence: {conf:.2f})")

            self._log(user_input, bot_msg)

    def toggle_personality(self):
        """Switch between casual and formal mode."""
        self.personality = "formal" if self.personality == "casual" else "casual"
        print(f"Personality mode set to: {self.personality}")

# =============================================================================
# Menu System
# =============================================================================
class Menu:
    """Terminal menu for interacting with Meemo."""

    def __init__(self, chat_session, ai_engine, db):
        self.chat = chat_session
        self.ai = ai_engine
        self.db = db

    def display(self):
        """Show menu options."""
        print("\n" + "="*50)
        print(" MEEmo BOT - MAIN MENU")
        print("="*50)
        print("1. Chat")
        print("2. Train manually (add Q&A)")
        print("3. View knowledge")
        print("4. Correct last response")
        print("5. Reset memory")
        print("6. Export database")
        print("7. Toggle personality (current: {})".format(self.chat.personality))
        print("8. Cluster knowledge (merge duplicates)")
        print("9. Inject knowledge from file")
        print("10. Inject knowledge (paste directly)")
        print("11. Exit")
        print("-"*50)
        print("Injection formats: JSON list with 'input'/'response' keys, or CSV with headers 'user_input,bot_response,weight'")
        print("Example JSON: [{\"input\":\"Hello\",\"response\":\"Hi!\"}]")

    def run(self):
        """Main menu loop."""
        while True:
            self.display()
            choice = input("Enter your choice: ").strip()

            if choice == '1':
                self.chat.chat()
            elif choice == '2':
                self._train_manual()
            elif choice == '3':
                self._view_knowledge()
            elif choice == '4':
                self._correct_last()
            elif choice == '5':
                self._reset_memory()
            elif choice == '6':
                self._export()
            elif choice == '7':
                self.chat.toggle_personality()
            elif choice == '8':
                self.ai.cluster_knowledge()
            elif choice == '9':
                self._inject_from_file()
            elif choice == '10':
                self.ai.inject_knowledge_paste()
            elif choice == '11':
                print("Goodbye from Meemo!")
                self.db.close()
                sys.exit(0)
            else:
                print("Invalid choice. Please try again.")

    def _train_manual(self):
        """Add a new Q&A pair manually."""
        print("\n--- Manual Training ---")
        q = input("Enter the user question: ").strip()
        if not q:
            print("Cancelled.")
            return
        a = input("Enter Meemo's answer: ").strip()
        if not a:
            print("Cancelled.")
            return
        self.ai.add_manual_entry(q, a)
        print("Training complete.")

    def _view_knowledge(self):
        """Display all stored knowledge."""
        entries = self.db.get_all_entries()
        if not entries:
            print("No knowledge yet.")
            return
        print("\n--- Knowledge Base ---")
        print(f"{'ID':<4} {'Weight':<6} {'Usage':<6} {'Question (truncated)':<30} {'Answer (truncated)'}")
        print("-"*80)
        for e in entries:
            q_short = e['user_input'][:27] + "..." if len(e['user_input']) > 30 else e['user_input']
            a_short = e['bot_response'][:27] + "..." if len(e['bot_response']) > 30 else e['bot_response']
            print(f"{e['id']:<4} {e['weight']:<6.2f} {e['usage_count']:<6} {q_short:<30} {a_short}")

    def _correct_last(self):
        """Correct the last response used in chat."""
        if self.ai.last_response_id is None:
            print("No last response recorded. Start a chat first.")
            return
        entry = self.db.get_entry_by_id(self.ai.last_response_id)
        if not entry:
            print("Last response no longer exists.")
            return
        print(f"Last response was to: '{entry['user_input']}'")
        print(f"Meemo said: '{entry['bot_response']}'")
        correct = input("Enter the correct answer: ").strip()
        if correct:
            self.ai.correct_last_response(correct)
        else:
            print("Cancelled.")

    def _reset_memory(self):
        """Delete all knowledge."""
        confirm = input("Are you sure you want to erase all memory? (yes/no): ").lower()
        if confirm == 'yes':
            self.db.delete_all()
            print("Memory reset.")
        else:
            print("Cancelled.")

    def _export(self):
        """Export database to CSV."""
        filename = input("Enter filename (default: meemo_export.csv): ").strip()
        if not filename:
            filename = "meemo_export.csv"
        self.db.export_to_csv(filename)

    def _inject_from_file(self):
        """Inject knowledge from a file."""
        print("\n--- Bulk Knowledge Injection from File ---")
        filepath = input("Enter path to the file: ").strip()
        if not filepath:
            print("Cancelled.")
            return
        self.ai.inject_knowledge_from_file(filepath)

# =============================================================================
# Main Application Entry Point
# =============================================================================
def main():
    """Initialize components and start the menu."""
    print("Starting Meemo Bot...")
    db = Database()
    processor = TextProcessor()
    ai = AIEngine(db, processor)
    chat = ChatSession(ai, db, processor)
    menu = Menu(chat, ai, db)
    menu.run()

if __name__ == "__main__":
    main()
