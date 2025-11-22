"""
Google Takeout Ingestion
=========================
Ingest Google Takeout data (Gmail, Search, Chrome, YouTube).
"""

import os
import glob
import json
import mailbox

from modules.core.types import NodeType, RelationType
from modules.ingestion.base import BaseIngestor


class GoogleTakeoutIngestor(BaseIngestor):
    """
    Ingest Google Takeout data (Gmail, Search, Chrome, YouTube).
    
    Supports:
    - Gmail (mbox format)
    - Search History (JSON)
    - Chrome History (JSON)
    - YouTube Watch History (JSON)
    """
    
    def ingest(self, takeout_dir: str):
        """
        Ingest Google Takeout data (Gmail, Search, Chrome, YouTube).
        
        Args:
            takeout_dir: Path to extracted Google Takeout folder
        """
        if not os.path.exists(takeout_dir):
            self.log(f"❌ Takeout path not found: {takeout_dir}")
            return
        
        self.log(f"📦 Processing Google Takeout from {takeout_dir}")
        
        # Gmail
        self._ingest_gmail(os.path.join(takeout_dir, "Mail"))
        
        # Search History
        self._ingest_search_history(os.path.join(takeout_dir, "My Activity"))
        
        # Chrome History
        self._ingest_chrome_history(os.path.join(takeout_dir, "Chrome"))
        
        # YouTube
        self._ingest_youtube(os.path.join(takeout_dir, "YouTube and YouTube Music"))
        
        self.log(f"✅ Takeout complete: {self.stats['emails']} emails, {self.stats['searches']} searches")
    
    def _ingest_gmail(self, mail_dir: str):
        """
        Parse Gmail mbox files.
        
        Args:
            mail_dir: Path to Mail directory
        """
        if not os.path.exists(mail_dir):
            return
        
        mbox_files = glob.glob(os.path.join(mail_dir, "*.mbox"))
        
        for mbox_file in mbox_files:
            try:
                mbox = mailbox.mbox(mbox_file)
                for message in mbox:
                    subject = message.get('subject', 'No Subject')
                    from_addr = message.get('from', '')
                    body = message.get_payload()
                    
                    if isinstance(body, list):
                        body = ' '.join([str(p.get_payload()) for p in body])
                    
                    # Create email node
                    email_id = f"email:{subject[:30]}"
                    content = f"Subject: {subject}\nFrom: {from_addr}\n{str(body)[:500]}"
                    
                    self.graph.add_node(email_id, NodeType.EVENT, content,
                                      metadata={'source': 'gmail', 'from': from_addr})
                    
                    self.stats['emails'] += 1
                    
                    # Extract entities from email
                    entities = self.extract_entities(content)
                    for ent_text, ent_type in entities[:5]:  # Limit per email
                        ent_id = f"{ent_type.value}:{ent_text}"
                        if ent_id not in self.graph.graph:
                            self.graph.add_node(ent_id, ent_type, ent_text)
                        self.graph.add_edge(email_id, ent_id, RelationType.MENTIONS)
                    
            except Exception as e:
                self.log(f"⚠️ Gmail mbox error: {str(e)[:50]}")
    
    def _ingest_search_history(self, activity_dir: str):
        """
        Parse Google Search history from My Activity.
        
        Args:
            activity_dir: Path to My Activity directory
        """
        if not os.path.exists(activity_dir):
            return
        
        search_files = glob.glob(os.path.join(activity_dir, "Search", "*.json"))
        
        for json_file in search_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    if 'title' in item and 'time' in item:
                        query = item['title'].replace("Searched for ", "")
                        timestamp = item['time']
                        
                        search_id = f"search:{query[:30]}"
                        self.graph.add_node(search_id, NodeType.EVENT, query,
                                          metadata={'source': 'google_search', 'time': timestamp})
                        
                        self.stats['searches'] += 1
                        
            except Exception as e:
                self.log(f"⚠️ Search history error: {str(e)[:50]}")
    
    def _ingest_chrome_history(self, chrome_dir: str):
        """
        Parse Chrome browsing history.
        
        Args:
            chrome_dir: Path to Chrome directory
        """
        if not os.path.exists(chrome_dir):
            return
        
        history_files = glob.glob(os.path.join(chrome_dir, "BrowserHistory.json"))
        
        for json_file in history_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'Browser History' in data:
                    for item in data['Browser History'][:500]:  # Limit to recent
                        url = item.get('url', '')
                        title = item.get('title', 'Untitled')
                        timestamp = item.get('time_usec', '')
                        
                        history_id = f"visit:{title[:30]}"
                        self.graph.add_node(history_id, NodeType.RESOURCE, url,
                                          metadata={'source': 'chrome', 'title': title, 'time': timestamp})
                        
                        self.stats['urls'] += 1
                        
            except Exception as e:
                self.log(f"⚠️ Chrome history error: {str(e)[:50]}")
    
    def _ingest_youtube(self, youtube_dir: str):
        """
        Parse YouTube watch history.
        
        Args:
            youtube_dir: Path to YouTube directory
        """
        if not os.path.exists(youtube_dir):
            return
        
        history_files = glob.glob(os.path.join(youtube_dir, "history", "watch-history.json"))
        
        for json_file in history_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data[:500]:  # Limit
                    title = item.get('title', 'Unknown')
                    timestamp = item.get('time', '')
                    
                    video_id = f"video:{title[:30]}"
                    self.graph.add_node(video_id, NodeType.RESOURCE, title,
                                      metadata={'source': 'youtube', 'time': timestamp})
                    
                    self.stats['videos'] += 1
                    
            except Exception as e:
                self.log(f"⚠️ YouTube history error: {str(e)[:50]}")
