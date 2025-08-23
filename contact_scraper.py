#!/usr/bin/env python3
"""
Railway Contact Scraper - Google Custom Search API Version
Reliable, fast contact scraping using Google Custom Search API
No more browser timeouts or HTML parsing issues!
"""

import asyncio
import time
import random
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from urllib.parse import quote, unquote
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import hashlib
from difflib import SequenceMatcher
from collections import defaultdict
import os
import tempfile
import requests
import sqlite3
from pathlib import Path

# API Framework
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Railway environment configuration
PORT = int(os.getenv("PORT", 8000))
RAILWAY_ENV = os.getenv("RAILWAY_ENVIRONMENT_NAME", "production")

# API Keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging for Railway
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchParameters:
    """Enhanced search parameters for Google API"""
    query: str
    industry: Optional[str] = None
    position: Optional[str] = None
    company: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    keywords: Optional[str] = None
    experience_level: Optional[str] = None
    company_size: Optional[str] = None
    max_results: int = 50
    min_confidence: float = 0.6
    use_ai_validation: bool = True
    enable_caching: bool = True
    
    def to_search_query(self) -> str:
        """Convert to optimized Google search query"""
        parts = []
        
        # Core search terms
        if self.query:
            parts.append(f'"{self.query}"')
        if self.position:
            parts.append(f'"{self.position}"')
        if self.company:
            parts.append(f'"{self.company}"')
        if self.industry:
            parts.append(f'"{self.industry}"')
            
        # Location
        if self.city and self.country:
            parts.append(f'"{self.city}, {self.country}"')
        elif self.country:
            parts.append(f'"{self.country}"')
        elif self.city:
            parts.append(f'"{self.city}"')
            
        # Additional keywords
        if self.keywords:
            parts.append(self.keywords)
        else:
            # Default to LinkedIn profiles
            parts.append('site:linkedin.com/in/')
            
        return " ".join(parts)

@dataclass  
class ContactResult:
    """Enhanced contact result with validation scores"""
    name: str
    position: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    industry: Optional[str] = None
    linkedin_url: Optional[str] = None
    company_website: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    experience_level: Optional[str] = None
    profile_summary: Optional[str] = None
    search_query: Optional[str] = None
    search_source: str = "Google Custom Search API"
    confidence_score: float = 0.0
    ai_validation_score: Optional[float] = None
    quality_notes: Optional[str] = None
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    cache_hit: bool = False
    
    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()

class ContactCache:
    """SQLite-based contact caching system"""
    
    def __init__(self, cache_file: str = "contacts_cache.db"):
        self.cache_file = cache_file
        self.init_cache()
    
    def init_cache(self):
        """Initialize cache database"""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_hash TEXT,
                    name TEXT,
                    position TEXT,
                    company TEXT,
                    location TEXT,
                    country TEXT,
                    industry TEXT,
                    linkedin_url TEXT,
                    company_website TEXT,
                    email TEXT,
                    phone TEXT,
                    experience_level TEXT,
                    profile_summary TEXT,
                    search_query TEXT,
                    confidence_score REAL,
                    ai_validation_score REAL,
                    quality_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(linkedin_url, search_hash)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_search_hash ON contacts(search_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON contacts(created_at)
            """)
            
            conn.commit()
            conn.close()
            logger.info("✅ Contact cache initialized")
            
        except Exception as e:
            logger.error(f"❌ Cache initialization failed: {e}")
    
    def get_search_hash(self, search_params: SearchParameters) -> str:
        """Generate hash for search parameters"""
        search_key = f"{search_params.query}|{search_params.position}|{search_params.company}|{search_params.industry}|{search_params.country}"
        return hashlib.md5(search_key.lower().encode()).hexdigest()
    
    def get_cached_contacts(self, search_params: SearchParameters, max_age_days: int = 7) -> List[ContactResult]:
        """Get cached contacts for search"""
        try:
            search_hash = self.get_search_hash(search_params)
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM contacts 
                WHERE search_hash = ? AND created_at > ?
                ORDER BY confidence_score DESC, ai_validation_score DESC
                LIMIT ?
            """, (search_hash, cutoff_date.isoformat(), search_params.max_results))
            
            rows = cursor.fetchall()
            conn.close()
            
            contacts = []
            for row in rows:
                contact = ContactResult(
                    name=row[2],
                    position=row[3],
                    company=row[4],
                    location=row[5],
                    country=row[6],
                    industry=row[7],
                    linkedin_url=row[8],
                    company_website=row[9],
                    email=row[10],
                    phone=row[11],
                    experience_level=row[12],
                    profile_summary=row[13],
                    search_query=row[14],
                    confidence_score=row[15] or 0.0,
                    ai_validation_score=row[16],
                    quality_notes=row[17],
                    scraped_at=row[18],
                    cache_hit=True
                )
                contacts.append(contact)
            
            logger.info(f"📦 Found {len(contacts)} cached contacts for search")
            return contacts
            
        except Exception as e:
            logger.error(f"❌ Cache retrieval failed: {e}")
            return []
    
    def cache_contacts(self, contacts: List[ContactResult], search_params: SearchParameters):
        """Cache new contacts"""
        try:
            search_hash = self.get_search_hash(search_params)
            
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            for contact in contacts:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO contacts 
                        (search_hash, name, position, company, location, country, industry,
                         linkedin_url, company_website, email, phone, experience_level,
                         profile_summary, search_query, confidence_score, ai_validation_score, quality_notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        search_hash,
                        contact.name,
                        contact.position,
                        contact.company,
                        contact.location,
                        contact.country,
                        contact.industry,
                        contact.linkedin_url,
                        contact.company_website,
                        contact.email,
                        contact.phone,
                        contact.experience_level,
                        contact.profile_summary,
                        contact.search_query,
                        contact.confidence_score,
                        contact.ai_validation_score,
                        contact.quality_notes
                    ))
                except sqlite3.IntegrityError:
                    # Contact already exists, skip
                    continue
            
            conn.commit()
            conn.close()
            logger.info(f"💾 Cached {len(contacts)} new contacts")
            
        except Exception as e:
            logger.error(f"❌ Caching failed: {e}")
    
    def cleanup_old_cache(self, max_age_days: int = 30):
        """Clean up old cached entries"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM contacts WHERE created_at < ?", (cutoff_date.isoformat(),))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"🧹 Cleaned up {deleted_count} old cache entries")
            
        except Exception as e:
            logger.error(f"❌ Cache cleanup failed: {e}")

class GoogleSearchClient:
    """Google Custom Search API client"""
    
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = None
        
    async def init_session(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = httpx.Timeout(30.0, connect=10.0)
            self.session = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    
    async def search(self, query: str, num_results: int = 10, start_index: int = 1) -> List[Dict]:
        """Perform Google Custom Search API call"""
        try:
            await self.init_session()
            
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),  # API limit is 10 per request
                'start': start_index
            }
            
            logger.info(f"🔍 Google API search: {query[:100]}...")
            
            response = await self.session.get(self.base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"❌ Google API error {response.status_code}: {response.text}")
                return []
            
            data = response.json()
            
            if 'error' in data:
                logger.error(f"❌ Google API error: {data['error']}")
                return []
            
            items = data.get('items', [])
            logger.info(f"✅ Google API returned {len(items)} results")
            
            return items
            
        except Exception as e:
            logger.error(f"❌ Google API search failed: {e}")
            return []
    
    async def batch_search(self, query: str, total_results: int = 50) -> List[Dict]:
        """Perform multiple API calls to get more results"""
        all_items = []
        
        # Calculate number of requests needed
        requests_needed = min((total_results + 9) // 10, 10)  # Max 10 requests (100 results)
        
        for i in range(requests_needed):
            start_index = i * 10 + 1
            
            items = await self.search(query, 10, start_index)
            
            if not items:
                break
                
            all_items.extend(items)
            
            # Respectful delay between requests
            if i < requests_needed - 1:
                await asyncio.sleep(0.5)
        
        return all_items[:total_results]
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.aclose()

class OpenAIValidator:
    """OpenAI-powered lead validation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        
    async def init_session(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = httpx.Timeout(60.0, connect=10.0)
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            self.session = httpx.AsyncClient(timeout=timeout, headers=headers)
    
    async def validate_contacts(self, contacts: List[ContactResult], search_params: SearchParameters) -> List[ContactResult]:
        """Batch validate contacts using OpenAI"""
        try:
            if not contacts:
                return []
                
            await self.init_session()
            
            # Prepare contacts for AI validation
            contacts_data = []
            for i, contact in enumerate(contacts):
                contacts_data.append({
                    'id': i + 1,
                    'name': contact.name or 'Unknown',
                    'position': contact.position or 'Not specified',
                    'company': contact.company or 'Not specified',
                    'location': contact.location or 'Not specified',
                    'linkedin_url': contact.linkedin_url or 'Not found',
                    'summary': contact.profile_summary or 'No summary'
                })
            
            # Create validation prompt
            prompt = f"""
You are an expert lead qualification specialist. Analyze these {len(contacts)} LinkedIn leads and validate their quality and relevance.

SEARCH CRITERIA:
- Query: "{search_params.query}"
- Position: {search_params.position or 'Any'}
- Company: {search_params.company or 'Any'}
- Industry: {search_params.industry or 'Any'}
- Location: {search_params.country or 'Any'}

LEADS TO VALIDATE:
{json.dumps(contacts_data, indent=2)}

VALIDATION CRITERIA:
1. Profile Quality: Complete and professional profile
2. Relevance: Matches search criteria
3. Authenticity: Real person, not fake profile
4. Professional Value: Appropriate for business outreach

Return ONLY valid JSON array with enhanced lead data:
[
  {{
    "id": 1,
    "name": "Enhanced Professional Name",
    "position": "Specific Job Title",
    "company": "Company Name",
    "location": "City, Country",
    "linkedin_url": "Profile URL",
    "validation_score": 8.5,
    "quality_notes": "Brief professional assessment",
    "industry": "Industry if determinable",
    "experience_level": "Entry/Mid/Senior/Executive",
    "email_likelihood": "High/Medium/Low",
    "contact_potential": "High/Medium/Low"
  }}
]

Return empty array [] if no leads meet professional standards (score < 7.0).
"""

            logger.info(f"🧠 AI validating {len(contacts)} contacts...")
            
            payload = {
                'model': 'gpt-4',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert lead qualification specialist. Always respond with valid JSON only.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 4000,
                'temperature': 0.2
            }
            
            response = await self.session.post(
                'https://api.openai.com/v1/chat/completions',
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"❌ OpenAI API error {response.status_code}")
                return contacts  # Return original contacts if AI fails
            
            result = response.json()
            
            if not result.get('choices'):
                logger.error("❌ Empty OpenAI response")
                return contacts
            
            ai_response = result['choices'][0]['message']['content'].strip()
            
            # Parse AI response
            try:
                # Clean response
                clean_response = ai_response.replace('```json\n', '').replace('\n```', '').strip()
                validated_data = json.loads(clean_response)
                
                if not isinstance(validated_data, list):
                    logger.error("❌ AI response not a list")
                    return contacts
                
                # Update original contacts with AI validation
                validated_contacts = []
                
                for ai_contact in validated_data:
                    original_idx = ai_contact.get('id', 1) - 1
                    
                    if 0 <= original_idx < len(contacts):
                        contact = contacts[original_idx]
                        
                        # Update with AI enhancements
                        contact.ai_validation_score = ai_contact.get('validation_score', contact.confidence_score)
                        contact.quality_notes = ai_contact.get('quality_notes', '')
                        contact.industry = ai_contact.get('industry') or contact.industry
                        contact.experience_level = ai_contact.get('experience_level') or contact.experience_level
                        
                        # Enhanced position and company from AI
                        if ai_contact.get('position') and ai_contact['position'] != 'Not specified':
                            contact.position = ai_contact['position']
                        if ai_contact.get('company') and ai_contact['company'] != 'Not specified':
                            contact.company = ai_contact['company']
                        
                        # Update overall confidence score
                        contact.confidence_score = max(contact.confidence_score, contact.ai_validation_score or 0)
                        
                        validated_contacts.append(contact)
                
                logger.info(f"✅ AI validated {len(validated_contacts)} high-quality contacts from {len(contacts)} total")
                return validated_contacts
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse AI response: {e}")
                return contacts
                
        except Exception as e:
            logger.error(f"❌ AI validation failed: {e}")
            return contacts  # Return original contacts if AI fails
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.aclose()

class RailwayContactScraperAPI:
    """Railway-optimized contact scraper using Google Custom Search API"""
    
    def __init__(self):
        self.google_client = None
        self.openai_validator = None
        self.cache = ContactCache()
        self.request_count = 0
        self.last_request_time = 0
        
        # Initialize API clients if keys available
        if GOOGLE_API_KEY and SEARCH_ENGINE_ID:
            self.google_client = GoogleSearchClient(GOOGLE_API_KEY, SEARCH_ENGINE_ID)
            logger.info("✅ Google Custom Search API initialized")
        else:
            logger.warning("⚠️ Google API credentials not found")
            
        if OPENAI_API_KEY:
            self.openai_validator = OpenAIValidator(OPENAI_API_KEY)
            logger.info("✅ OpenAI API initialized")
        else:
            logger.warning("⚠️ OpenAI API key not found - validation disabled")
    
    async def search_contacts(self, params: SearchParameters) -> List[ContactResult]:
        """Main contact search method"""
        logger.info(f"🔍 Starting contact search: {params.query}")
        
        # Check API availability
        if not self.google_client:
            raise Exception("Google Custom Search API not configured. Set GOOGLE_API_KEY and SEARCH_ENGINE_ID environment variables.")
        
        # Step 1: Check cache first
        cached_contacts = []
        if params.enable_caching:
            cached_contacts = self.cache.get_cached_contacts(params)
            
            if len(cached_contacts) >= params.max_results:
                logger.info(f"💨 Using {len(cached_contacts)} cached contacts - instant results!")
                return cached_contacts[:params.max_results]
        
        # Step 2: Search via Google API
        search_query = params.to_search_query()
        needed_results = params.max_results - len(cached_contacts)
        
        logger.info(f"🚀 Searching Google API for {needed_results} additional contacts...")
        search_results = await self.google_client.batch_search(search_query, needed_results * 2)
        
        # Step 3: Parse search results into contacts
        raw_contacts = []
        for item in search_results:
            contact = self._parse_search_result(item, params)
            if contact:
                raw_contacts.append(contact)
        
        # Step 4: Filter and deduplicate
        unique_contacts = self._deduplicate_contacts(raw_contacts, cached_contacts)
        
        # Step 5: AI validation (if enabled and API available)
        validated_contacts = unique_contacts
        if params.use_ai_validation and self.openai_validator and unique_contacts:
            validated_contacts = await self.openai_validator.validate_contacts(unique_contacts, params)
        
        # Step 6: Filter by confidence and sort
        final_contacts = [
            contact for contact in validated_contacts
            if contact.confidence_score >= params.min_confidence
        ]
        final_contacts.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Step 7: Cache new contacts
        if params.enable_caching and final_contacts:
            new_contacts = [c for c in final_contacts if not c.cache_hit]
            if new_contacts:
                self.cache.cache_contacts(new_contacts, params)
        
        # Step 8: Combine with cached results
        all_results = cached_contacts + final_contacts[:needed_results]
        final_results = all_results[:params.max_results]
        
        logger.info(f"✅ Search complete: {len(final_results)} contacts ({len(cached_contacts)} cached + {len(final_contacts)} new)")
        
        return final_results
    
    def _parse_search_result(self, item: Dict, params: SearchParameters) -> Optional[ContactResult]:
        """Parse Google search result into ContactResult"""
        try:
            title = item.get('title', '')
            link = item.get('link', '')
            snippet = item.get('snippet', '')
            
            # Skip non-LinkedIn results unless specifically requested
            if 'linkedin.com/in/' not in link and not params.keywords:
                return None
            
            # Extract name from LinkedIn URL or title
            name = self._extract_name_from_linkedin(link) or self._extract_name_from_title(title)
            if not name:
                return None
            
            # Extract other information
            position = self._extract_position(title, snippet, params.position)
            company = self._extract_company(title, snippet, params.company)
            location = self._extract_location(snippet, params.country, params.city)
            
            # Calculate base confidence
            confidence = self._calculate_confidence(name, position, company, location, params)
            
            if confidence < 0.3:  # Minimum threshold
                return None
            
            contact = ContactResult(
                name=name,
                position=position,
                company=company,
                location=location,
                country=self._extract_country(location, params.country),
                industry=params.industry,
                linkedin_url=link if 'linkedin.com' in link else None,
                profile_summary=snippet[:200] if snippet else None,
                search_query=params.query,
                confidence_score=confidence,
                experience_level=self._infer_experience_level(position, snippet)
            )
            
            return contact
            
        except Exception as e:
            logger.debug(f"⚠️ Failed to parse search result: {e}")
            return None
    
    def _extract_name_from_linkedin(self, url: str) -> Optional[str]:
        """Extract name from LinkedIn URL"""
        try:
            if not url or 'linkedin.com/in/' not in url:
                return None
            
            # Extract profile slug
            match = re.search(r'/in/([^/?]+)', url)
            if not match:
                return None
                
            slug = match.group(1)
            
            # Convert slug to readable name
            name_parts = slug.replace('-', ' ').split()
            
            if len(name_parts) >= 2:
                # Capitalize first two parts
                first_name = name_parts[0].capitalize()
                last_name = name_parts[1].capitalize()
                return f"{first_name} {last_name}"
            
            return None
            
        except Exception:
            return None
    
    def _extract_name_from_title(self, title: str) -> Optional[str]:
        """Extract name from search result title"""
        try:
            if not title:
                return None
            
            # Common patterns for LinkedIn titles
            patterns = [
                r'^([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})',  # First Last at start
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[-|–]',  # Name before dash
                r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)',  # First M. Last
            ]
            
            for pattern in patterns:
                match = re.search(pattern, title)
                if match:
                    name = match.group(1).strip()
                    if self._is_valid_name(name):
                        return name
            
            return None
            
        except Exception:
            return None
    
    def _extract_position(self, title: str, snippet: str, target_position: Optional[str] = None) -> Optional[str]:
        """Extract job position"""
        text = f"{title} {snippet}"
        
        # Check for target position first
        if target_position and target_position.lower() in text.lower():
            return target_position
        
        # Look for position patterns
        position_patterns = [
            r'(?:^|\s)([A-Z][a-zA-Z\s]{3,30})\s*(?:at|@|\|)',
            r'(?:CEO|CTO|CFO|VP|Director|Manager|Senior|Principal|Lead)\s+[A-Za-z\s]+',
            r'[A-Z][a-z]+\s+(?:Engineer|Developer|Manager|Director|Analyst|Consultant|Specialist)'
        ]
        
        for pattern in position_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                position = match.strip() if isinstance(match, str) else matches[0].strip()
                if len(position) > 3 and position.lower() not in ['linkedin', 'profile']:
                    return position
        
        return None
    
    def _extract_company(self, title: str, snippet: str, target_company: Optional[str] = None) -> Optional[str]:
        """Extract company name"""
        text = f"{title} {snippet}"
        
        # Check for target company first
        if target_company and target_company.lower() in text.lower():
            return target_company
        
        # Look for company patterns
        company_patterns = [
            r'\bat\s+([A-Z][A-Za-z\s&.]+(?:Inc|LLC|Corp|Ltd|Co)\.?)',
            r'([A-Z][A-Za-z\s&.]+(?:Inc|LLC|Corp|Ltd|Co)\.?)',
            r'\bat\s+([A-Z][A-Za-z\s&.]{3,25})',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                company = match.strip()
                if len(company) > 2 and company.lower() not in ['linkedin', 'profile', 'company']:
                    return company
        
        return None
    
    def _extract_location(self, snippet: str, target_country: Optional[str] = None, target_city: Optional[str] = None) -> Optional[str]:
        """Extract location information"""
        if not snippet:
            return None
        
        # Check for target location first
        if target_city and target_country:
            pattern = rf'\b{re.escape(target_city)}[,\s]*{re.escape(target_country)}\b'
            if re.search(pattern, snippet, re.IGNORECASE):
                return f"{target_city}, {target_country}"
        
        # Look for location patterns
        location_patterns = [
            r'\b([A-Z][a-z]+),\s*([A-Z]{2,})\b',
            r'(?:based|located|from)\s+in\s+([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)?)',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+),\s*([A-Z]{2,3})\b'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, snippet)
            if matches:
                if isinstance(matches[0], tuple):
                    return ', '.join(matches[0])
                return matches[0]
        
        return target_country or target_city
    
    def _extract_country(self, location: Optional[str], target_country: Optional[str] = None) -> Optional[str]:
        """Extract country from location"""
        if target_country:
            return target_country
        
        if location and ',' in location:
            parts = location.split(',')
            if len(parts) >= 2:
                return parts[-1].strip()
        
        return location
    
    def _infer_experience_level(self, position: Optional[str], snippet: str) -> Optional[str]:
        """Infer experience level from position and context"""
        if not position:
            return None
        
        position_lower = position.lower()
        
        if any(term in position_lower for term in ['ceo', 'cto', 'cfo', 'president', 'founder']):
            return 'Executive'
        elif any(term in position_lower for term in ['director', 'vp', 'vice president', 'head of']):
            return 'Director'
        elif any(term in position_lower for term in ['senior', 'lead', 'principal', 'staff']):
            return 'Senior'
        elif any(term in position_lower for term in ['manager', 'supervisor']):
            return 'Manager'
        elif any(term in position_lower for term in ['junior', 'entry', 'associate', 'trainee']):
            return 'Entry'
        else:
            return 'Mid'
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate if extracted text is likely a person's name"""
        if not name or len(name) < 3:
            return False
        
        # Check for obvious non-names
        invalid_terms = [
            'linkedin', 'profile', 'company', 'about', 'contact',
            'privacy', 'terms', 'policy', 'help', 'support'
        ]
        
        if any(term in name.lower() for term in invalid_terms):
            return False
        
        # Should have at least first and last name
        parts = name.split()
        if len(parts) < 2:
            return False
        
        # Each part should look like a name
        for part in parts:
            if not re.match(r'^[A-Z][a-z]{1,15}$', part):
                return False
        
        return True
    
    def _calculate_confidence(self, name: Optional[str], position: Optional[str], 
                            company: Optional[str], location: Optional[str], 
                            params: SearchParameters) -> float:
        """Calculate confidence score"""
        score = 0.0
        
        # Base score for having a name
        if name and self._is_valid_name(name):
            score += 0.4
        
        # Position matching
        if position:
            score += 0.2
            if params.position and params.position.lower() in position.lower():
                score += 0.1
        
        # Company matching  
        if company:
            score += 0.15
            if params.company and params.company.lower() in company.lower():
                score += 0.1
        
        # Location matching
        if location:
            score += 0.1
            if params.country and params.country.lower() in location.lower():
                score += 0.05
        
        # Data completeness
        data_fields = [name, position, company, location]
        completeness = sum(1 for field in data_fields if field) / len(data_fields)
        score += completeness * 0.1
        
        return min(score, 1.0)
    
    def _deduplicate_contacts(self, new_contacts: List[ContactResult], 
                            cached_contacts: List[ContactResult]) -> List[ContactResult]:
        """Remove duplicates between new and cached contacts"""
        if not new_contacts:
            return []
        
        # Create sets of existing identifiers
        cached_linkedin = {c.linkedin_url for c in cached_contacts if c.linkedin_url}
        cached_names = {c.name.lower() for c in cached_contacts if c.name}
        
        unique_contacts = []
        seen_linkedin = set()
        seen_names = set()
        
        for contact in new_contacts:
            # Skip if duplicate with cache
            if contact.linkedin_url and contact.linkedin_url in cached_linkedin:
                continue
                
            if contact.name and contact.name.lower() in cached_names:
                continue
            
            # Skip if duplicate within new contacts
            if contact.linkedin_url and contact.linkedin_url in seen_linkedin:
                continue
                
            name_key = contact.name.lower() if contact.name else None
            if name_key and name_key in seen_names:
                continue
            
            # Add to unique list
            unique_contacts.append(contact)
            
            if contact.linkedin_url:
                seen_linkedin.add(contact.linkedin_url)
            if name_key:
                seen_names.add(name_key)
        
        logger.info(f"🔄 Deduplication: {len(unique_contacts)} unique from {len(new_contacts)} new contacts")
        return unique_contacts
    
    async def cleanup_cache(self):
        """Clean up old cache entries"""
        self.cache.cleanup_old_cache()
    
    async def close(self):
        """Close all clients"""
        if self.google_client:
            await self.google_client.close()
        if self.openai_validator:
            await self.openai_validator.close()

# FastAPI Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Main search query")
    industry: Optional[str] = Field(None, description="Target industry")
    position: Optional[str] = Field(None, description="Job position/title")
    company: Optional[str] = Field(None, description="Company name")
    country: Optional[str] = Field(None, description="Country")
    city: Optional[str] = Field(None, description="City")
    keywords: Optional[str] = Field(None, description="Additional search keywords (e.g., 'site:company.com')")
    experience_level: Optional[str] = Field(None, description="Experience level")
    company_size: Optional[str] = Field(None, description="Company size")
    max_results: int = Field(50, ge=1, le=100, description="Maximum results")
    min_confidence: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence score")
    use_ai_validation: bool = Field(True, description="Enable AI validation")
    enable_caching: bool = Field(True, description="Enable smart caching")

class ContactResponse(BaseModel):
    name: str
    position: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    industry: Optional[str] = None
    linkedin_url: Optional[str] = None
    company_website: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    experience_level: Optional[str] = None
    profile_summary: Optional[str] = None
    search_query: Optional[str] = None
    search_source: str
    confidence_score: float
    ai_validation_score: Optional[float] = None
    quality_notes: Optional[str] = None
    scraped_at: str
    cache_hit: bool

class SearchResponse(BaseModel):
    success: bool
    message: str
    total_results: int
    cached_results: int
    new_results: int
    cache_hit_rate: float
    contacts: List[ContactResponse]
    search_params: dict
    ai_validation_used: bool
    data_quality: str

# FastAPI Application
app = FastAPI(
    title="Railway Contact Scraper - Google API Powered",
    description="Professional contact scraper using Google Custom Search API - No more timeouts!",
    version="2.0.0-GOOGLE-API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global scraper instance
scraper = RailwayContactScraperAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Railway Contact Scraper starting up with Google Custom Search API...")
    # Clean up old cache on startup
    await scraper.cleanup_cache()

@app.on_event("shutdown")
async def shutdown_event():
    await scraper.close()
    logger.info("👋 Railway Contact Scraper shutting down...")

@app.get("/")
async def root():
    return {
        "message": "Railway Contact Scraper - Google API Powered",
        "status": "healthy",
        "version": "2.0.0-GOOGLE-API",
        "architecture": "Google Custom Search API + OpenAI Validation",
        "features": [
            "✅ No more browser timeouts - uses Google API directly",
            "✅ Smart caching system with SQLite",
            "✅ OpenAI-powered lead validation",
            "✅ Advanced deduplication",
            "✅ Railway-optimized performance",
            "✅ Professional contact data only"
        ],
        "api_status": {
            "google_api_configured": scraper.google_client is not None,
            "openai_api_configured": scraper.openai_validator is not None,
            "caching_enabled": scraper.cache is not None
        },
        "search_examples": {
            "linkedin_professionals": {
                "query": "software engineer",
                "company": "Google",
                "keywords": "site:linkedin.com/in/"
            },
            "company_executives": {
                "query": "CEO startup",
                "industry": "technology",
                "country": "United States"
            },
            "specific_role": {
                "query": "marketing director",
                "position": "Marketing Director",
                "keywords": "site:linkedin.com/in/"
            }
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0-GOOGLE-API",
        "architecture": "Google Custom Search API",
        "google_api_ready": scraper.google_client is not None,
        "openai_api_ready": scraper.openai_validator is not None,
        "caching_ready": scraper.cache is not None,
        "request_count": scraper.request_count,
        "railway_optimized": True,
        "no_browser_needed": True,
        "reliability": "High - No timeout issues"
    }

@app.post("/search", response_model=SearchResponse)
async def search_contacts(request: SearchRequest):
    """Search for professional contacts using Google Custom Search API"""
    try:
        search_params = SearchParameters(**request.dict())
        
        # Perform the search
        results = await scraper.search_contacts(search_params)
        
        # Convert to response format
        contacts = [ContactResponse(**asdict(contact)) for contact in results]
        
        # Calculate metrics
        cached_count = sum(1 for contact in results if contact.cache_hit)
        new_count = len(results) - cached_count
        cache_hit_rate = (cached_count / len(results)) if results else 0.0
        
        # Determine data quality
        avg_confidence = sum(contact.confidence_score for contact in results) / len(results) if results else 0
        if avg_confidence >= 0.8:
            data_quality = "EXCELLENT - High confidence leads"
        elif avg_confidence >= 0.6:
            data_quality = "GOOD - Quality professional contacts"
        elif avg_confidence >= 0.4:
            data_quality = "FAIR - Some leads may need verification"
        else:
            data_quality = "LOW - Consider refining search criteria"
        
        return SearchResponse(
            success=True,
            message=f"Found {len(results)} professional contacts using Google Custom Search API",
            total_results=len(results),
            cached_results=cached_count,
            new_results=new_count,
            cache_hit_rate=cache_hit_rate,
            contacts=contacts,
            search_params=asdict(search_params),
            ai_validation_used=search_params.use_ai_validation and scraper.openai_validator is not None,
            data_quality=data_quality
        )
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    try:
        conn = sqlite3.connect(scraper.cache.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM contacts")
        total_contacts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT search_hash) FROM contacts")
        unique_searches = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM contacts 
            WHERE created_at > datetime('now', '-7 days')
        """)
        recent_contacts = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_cached_contacts": total_contacts,
            "unique_search_queries": unique_searches,
            "recent_contacts_7_days": recent_contacts,
            "cache_file": scraper.cache.cache_file,
            "cache_enabled": True
        }
        
    except Exception as e:
        logger.error(f"❌ Cache stats failed: {e}")
        return {
            "error": str(e),
            "cache_enabled": False
        }

@app.post("/cache/cleanup")
async def cleanup_cache():
    """Clean up old cache entries"""
    try:
        await scraper.cleanup_cache()
        return {
            "success": True,
            "message": "Cache cleanup completed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Cache cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache cleanup failed: {str(e)}")

@app.get("/api/config")
async def get_configuration():
    """Get current API configuration"""
    return {
        "google_custom_search": {
            "api_configured": GOOGLE_API_KEY is not None,
            "search_engine_configured": SEARCH_ENGINE_ID is not None,
            "status": "✅ Ready" if (GOOGLE_API_KEY and SEARCH_ENGINE_ID) else "❌ Not configured"
        },
        "openai_validation": {
            "api_configured": OPENAI_API_KEY is not None,
            "status": "✅ Ready" if OPENAI_API_KEY else "❌ Not configured - validation disabled"
        },
        "caching_system": {
            "enabled": True,
            "type": "SQLite",
            "status": "✅ Ready"
        },
        "architecture": "Google Custom Search API + OpenAI + SQLite Cache",
        "reliability": "High - No browser dependencies or timeout issues"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info"
    )
