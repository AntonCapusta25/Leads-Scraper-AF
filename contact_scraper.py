#!/usr/bin/env python3
"""
Professional Contact & People Scraper
Built with DrissionPage - Headless Mode
API-Ready for Frontend Integration
"""

import asyncio
import time
import random
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse, quote
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from difflib import SequenceMatcher
from collections import defaultdict
import glob        # Add this for wildcard path matching
import uuid        # Add this for random directories
# API Framework
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Railway environment configuration
import os
PORT = int(os.getenv("PORT", 8000))
RAILWAY_ENV = os.getenv("RAILWAY_ENVIRONMENT_NAME", "production")

# Configure logging for Railway
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchParameters:
    """Search parameters for contact scraping"""
    industry: Optional[str] = None
    position: Optional[str] = None
    company: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    keywords: Optional[str] = None
    experience_level: Optional[str] = None  # junior, senior, manager, director, etc.
    company_size: Optional[str] = None  # startup, small, medium, large, enterprise
    max_results: int = 50
    
    def to_search_string(self) -> str:
        """Convert parameters to search string"""
        parts = []
        if self.position: parts.append(f'"{self.position}"')
        if self.industry: parts.append(f'"{self.industry}"')
        if self.company: parts.append(f'"{self.company}"')
        if self.country: parts.append(self.country)
        if self.city: parts.append(self.city)
        if self.keywords: parts.append(self.keywords)
        return " ".join(parts)

@dataclass
class ContactResult:
    """Contact result data structure"""
    name: str
    position: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    profile_url: Optional[str] = None
    industry: Optional[str] = None
    experience: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = None
    scraped_at: Optional[str] = None
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()

class ContactScraper:
    """Professional Contact Scraper using DrissionPage"""
    
    def __init__(self, headless: bool = True, stealth: bool = True, dedup_strictness: str = "medium"):
        self.browser_page = None
        self.session_page = None
        self.headless = headless
        self.stealth = stealth
        self._browser_created = False
        
        # Deduplication settings
        self.dedup_strictness = dedup_strictness
        self.dedup_thresholds = self._get_dedup_thresholds(dedup_strictness)
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        # Rate limiting - more conservative for LinkedIn
        self.last_request_time = 0
        self.min_delay = 2  # Minimum delay between requests
        self.linkedin_delay = 5  # Longer delay for LinkedIn
        self.linkedin_request_count = 0
        self.linkedin_max_requests_per_hour = 20  # Very conservative
    
    def _get_dedup_thresholds(self, strictness: str) -> Dict[str, float]:
        """Get deduplication thresholds based on strictness level"""
        if strictness == "loose":
            return {
                "name_similarity": 0.7,
                "company_similarity": 0.6,
                "name_company_threshold": 0.65,
                "position_similarity": 0.6,
                "location_similarity": 0.5,
                "email_domain_name_threshold": 0.6
            }
        elif strictness == "strict":
            return {
                "name_similarity": 0.95,
                "company_similarity": 0.95,
                "name_company_threshold": 0.95,
                "position_similarity": 0.9,
                "location_similarity": 0.9,
                "email_domain_name_threshold": 0.9
            }
        else:  # medium (default)
            return {
                "name_similarity": 0.85,
                "company_similarity": 0.9,
                "name_company_threshold": 0.85,
                "position_similarity": 0.8,
                "location_similarity": 0.7,
                "email_domain_name_threshold": 0.8
            }
    
    def set_dedup_strictness(self, strictness: str):
        """Update deduplication strictness"""
        self.dedup_strictness = strictness
        self.dedup_thresholds = self._get_dedup_thresholds(strictness)
        logger.info(f"📊 Deduplication strictness set to: {strictness}")
    
    async def deduplicate_contacts(self, contacts: List[ContactResult], strictness: str = None) -> Tuple[List[ContactResult], Dict[str, Any]]:
        """
        Public method to deduplicate contacts with statistics
        """
        try:
            if strictness:
                old_strictness = self.dedup_strictness
                self.set_dedup_strictness(strictness)
            
            original_count = len(contacts)
            logger.info(f"🔄 Starting deduplication of {original_count} contacts with {self.dedup_strictness} strictness")
            
            # Perform deduplication
            deduplicated_contacts = self._deduplicate_results(contacts)
            
            # Generate statistics
            stats = self._generate_dedup_stats(contacts, deduplicated_contacts)
            
            if strictness:
                # Restore original strictness
                self.set_dedup_strictness(old_strictness)
            
            return deduplicated_contacts, stats
            
        except Exception as e:
            logger.error(f"❌ Contact deduplication failed: {e}")
            return contacts, {"error": str(e)}
    
    def _generate_dedup_stats(self, original: List[ContactResult], deduplicated: List[ContactResult]) -> Dict[str, Any]:
        """Generate deduplication statistics"""
        try:
            original_count = len(original)
            deduplicated_count = len(deduplicated)
            removed_count = original_count - deduplicated_count
            
            # Analyze what types of duplicates were found
            duplicate_types = {
                "email_duplicates": 0,
                "linkedin_duplicates": 0,
                "name_company_duplicates": 0,
                "phone_duplicates": 0,
                "fuzzy_matches": 0
            }
            
            # Count sources
            source_distribution = defaultdict(int)
            for contact in deduplicated:
                if contact.source:
                    source_distribution[contact.source] += 1
            
            # Confidence score distribution
            confidence_buckets = {"high": 0, "medium": 0, "low": 0}
            for contact in deduplicated:
                if contact.confidence_score >= 0.8:
                    confidence_buckets["high"] += 1
                elif contact.confidence_score >= 0.5:
                    confidence_buckets["medium"] += 1
                else:
                    confidence_buckets["low"] += 1
            
            return {
                "original_count": original_count,
                "deduplicated_count": deduplicated_count,
                "duplicates_removed": removed_count,
                "duplicate_rate": round((removed_count / original_count) * 100, 2) if original_count > 0 else 0,
                "strictness_used": self.dedup_strictness,
                "duplicate_types": duplicate_types,
                "source_distribution": dict(source_distribution),
                "confidence_distribution": confidence_buckets,
                "thresholds_used": self.dedup_thresholds
            }
            
        except Exception as e:
            logger.debug(f"⚠️ Stats generation failed: {e}")
            return {
                "original_count": len(original),
                "deduplicated_count": len(deduplicated),
                "error": str(e)
            }
        
    def _safe_import_drissionpage(self):
        """Safely import DrissionPage"""
        try:
            from DrissionPage import ChromiumPage, ChromiumOptions, SessionPage
            logger.info("✅ DrissionPage imported successfully")
            return ChromiumPage, ChromiumOptions, SessionPage, True
        except ImportError as e:
            logger.error(f"❌ DrissionPage not available: {e}")
            logger.error("Install with: pip install DrissionPage")
            return None, None, None, False
    
    def _create_stealth_options(self):
        """Create enhanced stealth browser options for Railway"""
        ChromiumPage, ChromiumOptions, SessionPage, available = self._safe_import_drissionpage()
        
        if not available:
            return None
            
        try:
            co = ChromiumOptions()
            
            # Railway Chrome detection
            chrome_found = False
            
            # Try environment variables first
            chrome_env_path = os.getenv('CHROME_BIN') or os.getenv('CHROMIUM_PATH')
            if chrome_env_path:
                # Handle wildcard paths
                if '*' in chrome_env_path:
                    import glob
                    matches = glob.glob(chrome_env_path)
                    if matches:
                        actual_path = matches[0]
                        if os.path.exists(actual_path):
                            co.set_browser_path(actual_path)
                            chrome_found = True
                            logger.info(f"✅ Found Chrome via env var: {actual_path}")
                elif os.path.exists(chrome_env_path):
                    co.set_browser_path(chrome_env_path)
                    chrome_found = True
                    logger.info(f"✅ Found Chrome via env var: {chrome_env_path}")
            
            # Try standard paths if env vars don't work
            if not chrome_found:
                chrome_paths = [
                    '/nix/store/*/bin/chromium',      # Railway/Nix
                    '/usr/bin/chromium',              # Standard
                    '/usr/bin/chromium-browser',      # Alternative
                    '/usr/bin/google-chrome-stable',  # Google Chrome
                ]
                
                for path in chrome_paths:
                    if '*' in path:
                        import glob
                        matches = glob.glob(path)
                        if matches:
                            actual_path = matches[0]
                            if os.path.exists(actual_path):
                                co.set_browser_path(actual_path)
                                chrome_found = True
                                logger.info(f"✅ Found Chrome: {actual_path}")
                                break
                    elif os.path.exists(path):
                        co.set_browser_path(path)
                        chrome_found = True
                        logger.info(f"✅ Found Chrome: {path}")
                        break
            
            if not chrome_found:
                logger.warning("⚠️ Chrome not found - letting DrissionPage auto-detect")
            
            # Railway-optimized arguments
            if self.headless:
                co.set_argument('--headless=new')
            
            railway_args = [
                '--no-sandbox',                    # Required for Railway
                '--disable-dev-shm-usage',         # Required for Railway
                '--disable-gpu',
                '--disable-web-security',
                '--disable-extensions',
                '--disable-plugins', 
                '--disable-images',
                '--no-first-run',
                '--disable-infobars',
                '--disable-notifications',
                '--disable-popup-blocking',
                '--disable-automation',
                '--disable-blink-features=AutomationControlled',
                '--window-size=1920,1080',
                '--user-agent=' + random.choice(self.user_agents)
            ]
            
            for arg in railway_args:
                co.set_argument(arg)
            
            # Set temp directory
            co.set_user_data_path(f'/tmp/chrome_{uuid.uuid4().hex[:8]}')
            
            return co
            
        except Exception as e:
            logger.error(f"❌ Chrome options failed: {e}")
            return None

    async def _create_browser(self):
        """Create browser instance with enhanced error handling"""
        try:
            if self._browser_created:
                return True
            
            ChromiumPage, ChromiumOptions, SessionPage, available = self._safe_import_drissionpage()
            
            if not available:
                logger.error("❌ DrissionPage not available - cannot create browser")
                return False
            
            # Create browser options
            options = self._create_stealth_options()
            if not options:
                logger.error("❌ Failed to create browser options")
                return False
            
            # Create browser page
            logger.info("🌐 Creating browser instance...")
            self.browser_page = ChromiumPage(addr_or_opts=options)
            
            # Test browser
            await asyncio.sleep(1)
            self.browser_page.get("https://httpbin.org/user-agent")
            await asyncio.sleep(2)
            
            self._browser_created = True
            logger.info("✅ Browser created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Browser creation failed: {e}")
            self.browser_page = None
            self._browser_created = False
            return False
    
    async def _rate_limit(self):
        """Apply standard rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last + random.uniform(0.5, 1.5)
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _rate_limit_linkedin(self):
        """Apply enhanced rate limiting for LinkedIn"""
        current_time = time.time()
        
        # Check hourly rate limit
        self.linkedin_request_count += 1
        if self.linkedin_request_count > self.linkedin_max_requests_per_hour:
            logger.warning("⚠️ LinkedIn hourly rate limit reached. Consider using LinkedIn API.")
            await asyncio.sleep(3600)  # Wait an hour
            self.linkedin_request_count = 0
        
        # Apply longer delay for LinkedIn
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.linkedin_delay:
            sleep_time = self.linkedin_delay - time_since_last + random.uniform(2, 5)
            logger.info(f"⏳ LinkedIn rate limiting: waiting {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def search_linkedin_api_alternative(self, params: SearchParameters) -> List[ContactResult]:
        """
        Alternative method using LinkedIn API (requires API key)
        This is the RECOMMENDED approach for production use
        """
        try:
            logger.info("📡 Using LinkedIn API (recommended method)")
            
            # This would require LinkedIn API credentials
            # Placeholder for LinkedIn API integration
            
            # Example API call structure:
            # headers = {
            #     'Authorization': f'Bearer {linkedin_api_token}',
            #     'Content-Type': 'application/json'
            # }
            # 
            # api_url = 'https://api.linkedin.com/v2/people-search'
            # response = requests.get(api_url, headers=headers, params=search_params)
            
            logger.warning("LinkedIn API integration not implemented. Add your API credentials.")
            return []
            
        except Exception as e:
            logger.error(f"❌ LinkedIn API failed: {e}")
            return []
    
    def _enhance_linkedin_profile_data(self, contact: ContactResult, profile_url: str) -> ContactResult:
        """Enhance contact data by analyzing LinkedIn profile URL"""
        try:
            # Extract LinkedIn username from URL
            username_match = re.search(r'/in/([^/?]+)', profile_url)
            if username_match:
                username = username_match.group(1)
                
                # Add additional metadata
                contact.linkedin_url = profile_url
                
                # Generate potential email patterns based on name and common patterns
                if contact.name and contact.company:
                    potential_emails = self._generate_email_patterns(contact.name, contact.company)
                    # You could validate these emails separately
                    contact.summary = f"Potential emails: {', '.join(potential_emails[:3])}"
            
            return contact
            
        except Exception as e:
            logger.debug(f"⚠️ LinkedIn profile enhancement failed: {e}")
            return contact
    
    def _generate_email_patterns(self, name: str, company: str) -> List[str]:
        """Generate common email patterns for a person/company"""
        try:
            if not name or not company:
                return []
            
            # Clean inputs
            name_parts = name.lower().replace('.', '').split()
            company_clean = re.sub(r'[^a-zA-Z0-9]', '', company.lower())
            
            if len(name_parts) < 2:
                return []
            
            first_name = name_parts[0]
            last_name = name_parts[-1]
            
            # Common email patterns
            patterns = [
                f"{first_name}.{last_name}@{company_clean}.com",
                f"{first_name}@{company_clean}.com",
                f"{last_name}@{company_clean}.com",
                f"{first_name[0]}{last_name}@{company_clean}.com",
                f"{first_name}{last_name[0]}@{company_clean}.com",
                f"{first_name}_{last_name}@{company_clean}.com",
                f"{first_name}{last_name}@{company_clean}.com"
            ]
            
            return patterns
            
        except Exception as e:
            logger.debug(f"⚠️ Email pattern generation failed: {e}")
            return []
    
    async def search_contacts(self, params: SearchParameters, enable_deduplication: bool = True) -> List[ContactResult]:
        """Main method to search for contacts with optional deduplication"""
        logger.info(f"🔍 Starting contact search with params: {params}")
        
        all_results = []
        
        # Search across multiple sources
        search_methods = [
            self._search_google_contacts,
            self._search_business_directories,
            self._search_company_websites,
            self._search_professional_networks
        ]
        
        for method in search_methods:
            try:
                # Apply appropriate rate limiting based on method
                if 'linkedin' in method.__name__.lower():
                    await self._rate_limit_linkedin()
                else:
                    await self._rate_limit()
                
                results = await method(params)
                all_results.extend(results)
                
                # Stop if we have enough results
                if len(all_results) >= params.max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Search method {method.__name__} failed: {e}")
                continue
        
        # Apply deduplication if enabled
        if enable_deduplication:
            unique_results, dedup_stats = await self.deduplicate_contacts(all_results)
            logger.info(f"📊 Deduplication stats: {dedup_stats.get('duplicates_removed', 0)} duplicates removed")
        else:
            unique_results = all_results
        
        # Sort by confidence and limit results
        sorted_results = sorted(unique_results, key=lambda x: x.confidence_score, reverse=True)
        final_results = sorted_results[:params.max_results]
        
        logger.info(f"✅ Found {len(final_results)} unique contacts")
        return final_results
    
    async def _search_google_contacts(self, params: SearchParameters) -> List[ContactResult]:
        """Search for contacts using Google search"""
        try:
            if not self.browser_page:
                await self._create_browser()
            
            # Build Google search query
            search_query = self._build_google_query(params)
            google_url = f"https://www.google.com/search?q={quote(search_query)}"
            
            logger.info(f"🔍 Google searching: {search_query}")
            
            self.browser_page.get(google_url)
            await asyncio.sleep(3)
            
            results = []
            
            # Parse Google results
            search_results = self.browser_page.eles('css:.g')
            
            for result in search_results[:10]:  # First 10 results
                try:
                    contact = self._parse_google_result(result, params)
                    if contact:
                        contact.source = "Google Search"
                        results.append(contact)
                except Exception as e:
                    logger.debug(f"⚠️ Failed to parse Google result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Google search failed: {e}")
            return []
    
    async def _search_business_directories(self, params: SearchParameters) -> List[ContactResult]:
        """Search business directories"""
        try:
            results = []
            
            # Example directories (you can add more)
            directories = [
                "yellowpages.com",
                "manta.com",
                "crunchbase.com",
                "apollo.io"
            ]
            
            for directory in directories:
                try:
                    await self._rate_limit()
                    directory_results = await self._search_directory(directory, params)
                    results.extend(directory_results)
                except Exception as e:
                    logger.debug(f"⚠️ Directory {directory} failed: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Business directory search failed: {e}")
            return []
    
    async def _search_company_websites(self, params: SearchParameters) -> List[ContactResult]:
        """Search company websites for contact information"""
        try:
            if not params.company:
                return []
            
            results = []
            
            # Find company website
            company_url = await self._find_company_website(params.company)
            
            if company_url:
                contacts = await self._scrape_company_contacts(company_url, params)
                results.extend(contacts)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Company website search failed: {e}")
            return []
    
    async def _search_professional_networks(self, params: SearchParameters) -> List[ContactResult]:
        """Search professional networks with LinkedIn support"""
        try:
            results = []
            
            # LinkedIn search (with proper ToS considerations)
            linkedin_results = await self._search_linkedin_carefully(params)
            results.extend(linkedin_results)
            
            # Other professional networks can be added here
            # AngelList, Xing, etc.
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Professional network search failed: {e}")
            return []
    
    async def _search_linkedin_carefully(self, params: SearchParameters) -> List[ContactResult]:
        """
        LinkedIn search with extreme caution and ToS compliance
        WARNING: LinkedIn has strict ToS. Use responsibly and consider their API instead.
        """
        try:
            # IMPORTANT: This should only be used for legitimate business purposes
            # and in compliance with LinkedIn's Terms of Service
            logger.warning("⚠️ LinkedIn scraping - ensure ToS compliance!")
            
            if not self.browser_page:
                await self._create_browser()
            
            results = []
            
            # Build LinkedIn search URL (public search only)
            search_query = self._build_linkedin_search_query(params)
            
            # Use Google to find LinkedIn profiles (more ToS compliant)
            google_linkedin_results = await self._search_linkedin_via_google(params)
            results.extend(google_linkedin_results)
            
            # Direct LinkedIn search (use with extreme caution)
            if len(results) < params.max_results // 2:  # Only if not enough results
                direct_results = await self._search_linkedin_direct(params)
                results.extend(direct_results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ LinkedIn search failed: {e}")
            return []
    
    async def _search_linkedin_via_google(self, params: SearchParameters) -> List[ContactResult]:
        """Search LinkedIn profiles via Google (more ToS compliant)"""
        try:
            if not self.browser_page:
                await self._create_browser()
            
            # Build Google search for LinkedIn profiles
            search_terms = []
            
            if params.position:
                search_terms.append(f'"{params.position}"')
            if params.company:
                search_terms.append(f'"{params.company}"')
            if params.industry:
                search_terms.append(f'"{params.industry}"')
            if params.city or params.country:
                location = f"{params.city or ''} {params.country or ''}".strip()
                search_terms.append(f'"{location}"')
            
            search_query = " ".join(search_terms) + " site:linkedin.com/in/"
            
            google_url = f"https://www.google.com/search?q={quote(search_query)}&num=20"
            
            logger.info(f"🔍 Searching LinkedIn via Google: {search_query}")
            
            await self._rate_limit_linkedin()
            self.browser_page.get(google_url)
            await asyncio.sleep(3)
            
            results = []
            search_results = self.browser_page.eles('css:.g')
            
            for result in search_results[:10]:
                try:
                    link_elem = result.ele('css:a', timeout=1)
                    title_elem = result.ele('css:h3', timeout=1)
                    snippet_elem = result.ele('css:.VwiC3b', timeout=1)
                    
                    if not link_elem or not title_elem:
                        continue
                    
                    url = link_elem.attr('href')
                    title = title_elem.text or ""
                    snippet = snippet_elem.text if snippet_elem else ""
                    
                    # Verify it's a LinkedIn profile URL
                    if 'linkedin.com/in/' not in url:
                        continue
                    
                    # Parse profile information from search result
                    contact = self._parse_linkedin_search_result(title, snippet, url, params)
                    if contact:
                        # Enhance with additional LinkedIn data
                        contact = self._enhance_linkedin_profile_data(contact, url)
                        results.append(contact)
                        
                except Exception as e:
                    logger.debug(f"⚠️ Failed to parse LinkedIn search result: {e}")
                    continue
            
            logger.info(f"✅ Found {len(results)} LinkedIn profiles via Google")
            return results
            
        except Exception as e:
            logger.error(f"❌ LinkedIn Google search failed: {e}")
            return []
    
    async def _search_linkedin_direct(self, params: SearchParameters) -> List[ContactResult]:
        """
        Direct LinkedIn search - USE WITH EXTREME CAUTION
        WARNING: This may violate LinkedIn ToS. Consider using their API instead.
        """
        try:
            logger.warning("⚠️ CAUTION: Direct LinkedIn search - ensure compliance!")
            
            # This is a very basic implementation
            # In production, you should use LinkedIn's official API
            
            if not self.browser_page:
                await self._create_browser()
            
            # Build LinkedIn search URL
            search_params_dict = {}
            
            if params.position:
                search_params_dict['keywords'] = params.position
            if params.company:
                search_params_dict['currentCompany'] = params.company
            if params.city or params.country:
                location = f"{params.city or ''} {params.country or ''}".strip()
                search_params_dict['geoUrn'] = location
            
            # Convert to URL parameters
            url_params = "&".join([f"{k}={quote(str(v))}" for k, v in search_params_dict.items()])
            linkedin_search_url = f"https://www.linkedin.com/search/results/people/?{url_params}"
            
            logger.info(f"🔍 Direct LinkedIn search: {linkedin_search_url}")
            
            # Apply extra delay for LinkedIn
            await asyncio.sleep(random.uniform(3, 6))
            
            self.browser_page.get(linkedin_search_url)
            await asyncio.sleep(5)  # Wait for page load
            
            # Check if we're blocked or need to login
            page_content = self.browser_page.html.lower()
            if 'sign in' in page_content or 'join linkedin' in page_content:
                logger.warning("⚠️ LinkedIn requires authentication - switching to public search")
                return []
            
            results = []
            
            # Parse LinkedIn search results (this is a basic example)
            profile_cards = self.browser_page.eles('css:.entity-result__item')
            
            for card in profile_cards[:5]:  # Limit to prevent blocking
                try:
                    await asyncio.sleep(random.uniform(1, 2))  # Rate limiting
                    
                    contact = await self._parse_linkedin_profile_card(card, params)
                    if contact:
                        results.append(contact)
                        
                except Exception as e:
                    logger.debug(f"⚠️ Failed to parse LinkedIn profile card: {e}")
                    continue
            
            logger.info(f"✅ Found {len(results)} LinkedIn profiles directly")
            return results
            
        except Exception as e:
            logger.error(f"❌ Direct LinkedIn search failed: {e}")
            return []
    
    def _parse_linkedin_search_result(self, title: str, snippet: str, url: str, params: SearchParameters) -> Optional[ContactResult]:
        """Parse LinkedIn profile from Google search result"""
        try:
            # Extract name from title (usually "FirstName LastName - Position at Company")
            name_match = re.match(r'^([^-|]+?)(?:\s*[-|]|$)', title)
            name = name_match.group(1).strip() if name_match else "Unknown"
            
            # Extract position and company from title
            position_company_match = re.search(r'-\s*(.+?)\s*(?:at|@)\s*(.+?)(?:\s*[-|]|$)', title)
            if position_company_match:
                position = position_company_match.group(1).strip()
                company = position_company_match.group(2).strip()
            else:
                # Try alternative patterns
                position = self._extract_position_from_text(title + " " + snippet, params.position)
                company = self._extract_company_from_text(title + " " + snippet, params.company)
            
            # Extract location from snippet
            location = self._extract_location_from_text(snippet, params.country, params.city)
            
            # Calculate confidence score
            confidence = self._calculate_linkedin_confidence(name, position, company, params)
            
            if confidence > 0.4:  # Higher threshold for LinkedIn
                return ContactResult(
                    name=name,
                    position=position,
                    company=company,
                    location=location,
                    linkedin_url=url,
                    profile_url=url,
                    source="LinkedIn (via Google)",
                    confidence_score=confidence,
                    summary=snippet[:200] if snippet else None
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Failed to parse LinkedIn search result: {e}")
            return None
    
    async def _parse_linkedin_profile_card(self, card_element, params: SearchParameters) -> Optional[ContactResult]:
        """Parse LinkedIn profile card element"""
        try:
            # Extract name
            name_elem = card_element.ele('css:.entity-result__title-text a span[aria-hidden="true"]', timeout=2)
            name = name_elem.text.strip() if name_elem and name_elem.text else "Unknown"
            
            # Extract profile URL
            profile_link_elem = card_element.ele('css:.entity-result__title-text a', timeout=2)
            profile_url = profile_link_elem.attr('href') if profile_link_elem else ""
            
            # Extract position
            position_elem = card_element.ele('css:.entity-result__primary-subtitle', timeout=2)
            position = position_elem.text.strip() if position_elem and position_elem.text else None
            
            # Extract company and location
            secondary_elem = card_element.ele('css:.entity-result__secondary-subtitle', timeout=2)
            secondary_text = secondary_elem.text if secondary_elem and secondary_elem.text else ""
            
            # Parse company and location from secondary text
            company = None
            location = None
            
            if secondary_text:
                # Usually in format "Company • Location" or just "Company"
                parts = secondary_text.split('•')
                if len(parts) >= 2:
                    company = parts[0].strip()
                    location = parts[1].strip()
                elif len(parts) == 1:
                    # Could be company or location
                    if any(keyword in parts[0].lower() for keyword in ['inc', 'corp', 'ltd', 'llc', 'company']):
                        company = parts[0].strip()
                    else:
                        location = parts[0].strip()
            
            # Calculate confidence
            confidence = self._calculate_linkedin_confidence(name, position, company, params)
            
            if confidence > 0.4:
                return ContactResult(
                    name=name,
                    position=position,
                    company=company,
                    location=location,
                    linkedin_url=profile_url,
                    profile_url=profile_url,
                    source="LinkedIn (Direct)",
                    confidence_score=confidence
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Failed to parse LinkedIn profile card: {e}")
            return None
    
    def _calculate_linkedin_confidence(self, name: Optional[str], position: Optional[str], 
                                     company: Optional[str], params: SearchParameters) -> float:
        """Calculate confidence score for LinkedIn profiles"""
        try:
            score = 0.0
            
            # Base score for having a name
            if name and name != "Unknown":
                score += 0.4  # Higher base for LinkedIn
            
            # Position match (higher weight for LinkedIn)
            if position and params.position:
                if params.position.lower() in position.lower():
                    score += 0.4
                else:
                    score += 0.1
            elif position:
                score += 0.1
            
            # Company match (higher weight)
            if company and params.company:
                if params.company.lower() in company.lower():
                    score += 0.3
                else:
                    score += 0.1
            elif company:
                score += 0.1
            
            # LinkedIn profiles are generally more reliable
            score += 0.1
            
            return min(score, 1.0)
            
        except:
            return 0.0
    
    def _build_linkedin_search_query(self, params: SearchParameters) -> str:
        """Build LinkedIn search query"""
        try:
            parts = []
            
            if params.position:
                parts.append(f'title:"{params.position}"')
            if params.company:
                parts.append(f'company:"{params.company}"')
            if params.industry:
                parts.append(f'industry:"{params.industry}"')
            if params.city or params.country:
                location = f"{params.city or ''} {params.country or ''}".strip()
                parts.append(f'location:"{location}"')
            
            return " AND ".join(parts) if parts else ""
            
        except:
            return ""
    
    def _build_google_query(self, params: SearchParameters) -> str:
        """Build optimized Google search query"""
        query_parts = []
        
        # Core search terms
        if params.position and params.company:
            query_parts.append(f'"{params.position}" "{params.company}"')
        elif params.position:
            query_parts.append(f'"{params.position}"')
        elif params.company:
            query_parts.append(f'"{params.company}"')
        elif params.industry:
            query_parts.append(f'"{params.industry}"')
        
        # If no core terms, add default professional search terms
        if not query_parts:
            query_parts.append('(CEO OR director OR manager OR "contact us" OR "leadership team")')
        
        # Location
        if params.city and params.country:
            query_parts.append(f'"{params.city}, {params.country}"')
        elif params.country:
            query_parts.append(f'"{params.country}"')
        elif params.city:
            query_parts.append(f'"{params.city}"')
        
        # Industry
        if params.industry and params.industry not in " ".join(query_parts):
            query_parts.append(f'"{params.industry}"')
        
        # Add contact-specific terms
        query_parts.append('(email OR contact OR linkedin OR "about us" OR team)')
        
        # Exclude job boards and generic sites
        query_parts.append('-indeed.com -glassdoor.com -jobsite.com -linkedin.com/jobs')
        
        return " ".join(query_parts)
    
    def _parse_google_result(self, result_element, params: SearchParameters) -> Optional[ContactResult]:
        """Parse individual Google search result"""
        try:
            # Extract title, link, and snippet
            title_elem = result_element.ele('css:h3', timeout=1)
            link_elem = result_element.ele('css:a', timeout=1)
            snippet_elem = result_element.ele('css:.VwiC3b', timeout=1)
            
            if not title_elem or not link_elem:
                return None
            
            title = title_elem.text or ""
            url = link_elem.attr('href') or ""
            snippet = snippet_elem.text if snippet_elem else ""
            
            # Skip unwanted domains
            excluded_domains = ['indeed.com', 'glassdoor.com', 'jobsite.com', 'wikipedia.org']
            if any(domain in url.lower() for domain in excluded_domains):
                return None
            
            # Extract information from title and snippet
            full_text = f"{title} {snippet}"
            name = self._extract_name_from_text(title)
            position = self._extract_position_from_text(full_text, params.position)
            company = self._extract_company_from_text(full_text, params.company)
            location = self._extract_location_from_text(snippet, params.country, params.city)
            
            # Try to find email pattern
            email = self._extract_email_from_text(snippet)
            
            # If no name found but we have other good data, create a generic entry
            if not name and (email or (company and position)):
                if email:
                    name = self._extract_name_from_email(email)
                else:
                    name = f"{position or 'Professional'} at {company or 'Company'}"
            
            # Calculate confidence score
            confidence = self._calculate_confidence(name, position, company, params)
            
            # Lower threshold for location-only searches
            min_confidence = 0.2 if not params.position and not params.company else 0.3
            
            if confidence > min_confidence and name:
                return ContactResult(
                    name=name,
                    position=position,
                    company=company,
                    location=location,
                    email=email,
                    profile_url=url,
                    confidence_score=confidence,
                    summary=snippet[:200] if snippet else None
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Failed to parse Google result: {e}")
            return None
    
    async def _search_directory(self, directory: str, params: SearchParameters) -> List[ContactResult]:
        """Search specific business directory"""
        try:
            # This is a template - implement specific directory parsing
            logger.info(f"Searching {directory}")
            
            # Build directory-specific search URL
            search_url = self._build_directory_url(directory, params)
            
            if not search_url:
                return []
            
            if not self.browser_page:
                await self._create_browser()
            
            self.browser_page.get(search_url)
            await asyncio.sleep(3)
            
            # Directory-specific parsing would go here
            # This is a placeholder
            return []
            
        except Exception as e:
            logger.debug(f"⚠️ Directory search failed for {directory}: {e}")
            return []
    
    def _build_directory_url(self, directory: str, params: SearchParameters) -> str:
        """Build directory-specific search URL"""
        # Implement directory-specific URL building
        base_urls = {
            "yellowpages.com": "https://www.yellowpages.com/search?search_terms={}&geo_location_terms={}",
            "manta.com": "https://www.manta.com/search?search={}&location={}",
            "crunchbase.com": "https://www.crunchbase.com/discover/organization.companies/{}",
            "apollo.io": "https://app.apollo.io/#/people?{}"
        }
        
        if directory in base_urls:
            search_term = params.to_search_string()
            location = f"{params.city or ''} {params.country or ''}".strip()
            return base_urls[directory].format(quote(search_term), quote(location))
        
        return ""
    
    async def _find_company_website(self, company_name: str) -> Optional[str]:
        """Find company website URL"""
        try:
            if not self.browser_page:
                await self._create_browser()
            
            # Search for company website
            search_query = f'"{company_name}" site:official OR site:company OR website'
            google_url = f"https://www.google.com/search?q={quote(search_query)}"
            
            self.browser_page.get(google_url)
            await asyncio.sleep(2)
            
            # Get first result that looks like company website
            results = self.browser_page.eles('css:.g a')
            
            for result in results[:5]:
                try:
                    url = result.attr('href')
                    if url and self._is_company_website(url, company_name):
                        return url
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Company website search failed: {e}")
            return None
    
    def _is_company_website(self, url: str, company_name: str) -> bool:
        """Check if URL is likely a company website"""
        try:
            domain = urlparse(url).netloc.lower()
            company_clean = re.sub(r'[^a-zA-Z0-9]', '', company_name.lower())
            
            # Check if company name is in domain
            if company_clean in domain.replace('.', '').replace('-', ''):
                return True
            
            # Exclude common non-company domains
            excluded_domains = [
                'linkedin.com', 'facebook.com', 'twitter.com', 'instagram.com',
                'wikipedia.org', 'crunchbase.com', 'bloomberg.com', 'reuters.com',
                'google.com', 'youtube.com', 'indeed.com', 'glassdoor.com'
            ]
            
            for excluded in excluded_domains:
                if excluded in domain:
                    return False
            
            return True
            
        except:
            return False
    
    async def _scrape_company_contacts(self, company_url: str, params: SearchParameters) -> List[ContactResult]:
        """Scrape contacts from company website"""
        try:
            if not self.browser_page:
                await self._create_browser()
            
            results = []
            
            # Common contact pages
            contact_pages = [
                company_url,
                urljoin(company_url, '/about'),
                urljoin(company_url, '/team'),
                urljoin(company_url, '/contact'),
                urljoin(company_url, '/leadership'),
                urljoin(company_url, '/management')
            ]
            
            for page_url in contact_pages:
                try:
                    await self._rate_limit()
                    
                    self.browser_page.get(page_url)
                    await asyncio.sleep(2)
                    
                    page_contacts = self._extract_contacts_from_page(
                        self.browser_page.html, 
                        company_url, 
                        params
                    )
                    results.extend(page_contacts)
                    
                except Exception as e:
                    logger.debug(f"⚠️ Failed to scrape {page_url}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Company contact scraping failed: {e}")
            return []
    
    def _extract_contacts_from_page(self, html: str, base_url: str, params: SearchParameters) -> List[ContactResult]:
        """Extract contacts from HTML page"""
        try:
            contacts = []
            
            # Find email addresses
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', html)
            
            # Find phone numbers
            phones = re.findall(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', html)
            
            # Find potential names and positions (basic implementation)
            # This would need more sophisticated NLP for better results
            name_position_patterns = [
                r'<h[1-6][^>]*>([^<]+)</h[1-6]>',
                r'<p[^>]*><strong>([^<]+)</strong>',
                r'<div[^>]*class="[^"]*name[^"]*"[^>]*>([^<]+)</div>',
                r'<span[^>]*class="[^"]*title[^"]*"[^>]*>([^<]+)</span>'
            ]
            
            potential_contacts = []
            
            for pattern in name_position_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                potential_contacts.extend(matches)
            
            # Process found information
            for i, email in enumerate(emails[:5]):  # Limit to first 5 emails
                name = self._extract_name_from_email(email)
                
                contact = ContactResult(
                    name=name,
                    email=email,
                    company=params.company,
                    source=f"Company Website: {base_url}",
                    confidence_score=0.7  # Medium confidence for website contacts
                )
                
                contacts.append(contact)
            
            return contacts
            
        except Exception as e:
            logger.debug(f"⚠️ Contact extraction failed: {e}")
            return []
    
    # Utility methods for text extraction
    def _extract_name_from_text(self, text: str) -> Optional[str]:
        """Extract person name from text"""
        try:
            if not text:
                return None
            
            # Enhanced name extraction patterns
            name_patterns = [
                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # First Last
                r'\b([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)\b',  # First M. Last
                r'\b([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)\b',  # First Middle Last
                r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b',  # First Last-Name
                r'(?:CEO|Director|Manager|President)\s+([A-Z][a-z]+ [A-Z][a-z]+)',  # Title Name
                r'([A-Z][a-z]+ [A-Z][a-z]+)(?:\s*,\s*(?:CEO|Director|Manager|President))',  # Name, Title
            ]
            
            for pattern in name_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    # Return first valid name (filter out common non-names)
                    for match in matches:
                        name = match if isinstance(match, str) else match[0] if isinstance(match, tuple) else str(match)
                        # Filter out common false positives
                        if not any(word in name.lower() for word in ['about', 'contact', 'services', 'company', 'group', 'team']):
                            return name
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Name extraction failed: {e}")
            return None
    
    def _extract_position_from_text(self, text: str, target_position: Optional[str] = None) -> Optional[str]:
        """Extract job position from text"""
        try:
            # Common position keywords
            position_patterns = [
                r'\b(CEO|CTO|CFO|COO|VP|Director|Manager|Lead|Senior|Principal|Head of)\b[^,\.]{0,50}',
                r'\b(President|Founder|Co-founder|Partner|Executive|Analyst|Specialist|Engineer|Developer)\b[^,\.]{0,30}'
            ]
            
            for pattern in position_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[0]
            
            # If target position provided, look for it specifically
            if target_position:
                pattern = rf'\b{re.escape(target_position)}\b'
                if re.search(pattern, text, re.IGNORECASE):
                    return target_position
            
            return None
            
        except:
            return None
    
    def _extract_company_from_text(self, text: str, target_company: Optional[str] = None) -> Optional[str]:
        """Extract company name from text"""
        try:
            # If target company provided, look for it
            if target_company:
                pattern = rf'\b{re.escape(target_company)}\b'
                if re.search(pattern, text, re.IGNORECASE):
                    return target_company
            
            # Look for company indicators
            company_patterns = [
                r'\bat\s+([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Ltd|Co)\.?)\b',
                r'\b([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Ltd|Co)\.?)\b',
                r'\bworks?\s+at\s+([A-Z][a-zA-Z\s&]+)\b'
            ]
            
            for pattern in company_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[0].strip()
            
            return None
            
        except:
            return None
    
    def _extract_location_from_text(self, text: str, target_country: Optional[str] = None, target_city: Optional[str] = None) -> Optional[str]:
        """Extract location from text"""
        try:
            # Check for target location first
            if target_city and target_country:
                pattern = rf'\b{re.escape(target_city)}[,\s]+{re.escape(target_country)}\b'
                if re.search(pattern, text, re.IGNORECASE):
                    return f"{target_city}, {target_country}"
            
            # Common location patterns
            location_patterns = [
                r'\b([A-Z][a-z]+,\s*[A-Z][a-z]+)\b',  # City, Country
                r'\b([A-Z][a-z]+,\s*[A-Z]{2})\b',     # City, State
                r'\bin\s+([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)?)\b'  # in Location
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    return matches[0]
            
            return None
            
        except:
            return None
    
    def _extract_email_from_text(self, text: str) -> Optional[str]:
        """Extract email from text"""
        try:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            matches = re.findall(email_pattern, text)
            return matches[0] if matches else None
        except:
            return None
    
    def _extract_name_from_email(self, email: str) -> str:
        """Extract name from email address"""
        try:
            username = email.split('@')[0]
            
            # Handle common email patterns
            if '.' in username:
                parts = username.split('.')
                name_parts = [part.capitalize() for part in parts if len(part) > 1]
                return ' '.join(name_parts)
            elif '_' in username:
                parts = username.split('_')
                name_parts = [part.capitalize() for part in parts if len(part) > 1]
                return ' '.join(name_parts)
            else:
                return username.capitalize()
                
        except:
            return "Unknown"
    
    def _calculate_confidence(self, name: Optional[str], position: Optional[str], 
                            company: Optional[str], params: SearchParameters) -> float:
        """Calculate confidence score for a contact"""
        try:
            score = 0.0
            
            # Base score for having a name
            if name and name != "Unknown":
                score += 0.3
            
            # Position match
            if position and params.position:
                if params.position.lower() in position.lower():
                    score += 0.3
                else:
                    score += 0.1
            elif position:
                score += 0.1
            
            # Company match
            if company and params.company:
                if params.company.lower() in company.lower():
                    score += 0.3
                else:
                    score += 0.1
            elif company:
                score += 0.1
            
            # Additional factors
            if name and len(name.split()) >= 2:  # Full name
                score += 0.1
            
            return min(score, 1.0)  # Cap at 1.0
            
        except:
            return 0.0
    
    def _deduplicate_results(self, results: List[ContactResult]) -> List[ContactResult]:
        """
        Advanced deduplication with fuzzy matching and data merging
        """
        try:
            if not results:
                return []
            
            logger.info(f"🔄 Starting deduplication for {len(results)} contacts...")
            
            # Step 1: Group potentially duplicate contacts
            duplicate_groups = self._group_duplicates(results)
            
            # Step 2: Merge duplicates within each group
            deduplicated_contacts = []
            for group in duplicate_groups:
                if len(group) == 1:
                    deduplicated_contacts.append(group[0])
                else:
                    # Merge multiple contacts into one
                    merged_contact = self._merge_contacts(group)
                    deduplicated_contacts.append(merged_contact)
            
            # Step 3: Final validation and cleanup
            final_contacts = self._final_dedup_validation(deduplicated_contacts)
            
            removed_count = len(results) - len(final_contacts)
            logger.info(f"✅ Deduplication complete: {removed_count} duplicates removed, {len(final_contacts)} unique contacts")
            
            return final_contacts
            
        except Exception as e:
            logger.error(f"❌ Deduplication failed: {e}")
            # Fallback to basic deduplication
            return self._basic_deduplicate(results)
    
    def _group_duplicates(self, results: List[ContactResult]) -> List[List[ContactResult]]:
        """Group contacts that are likely duplicates"""
        try:
            groups = []
            processed = set()
            
            for i, contact in enumerate(results):
                if i in processed:
                    continue
                
                # Start a new group with this contact
                current_group = [contact]
                processed.add(i)
                
                # Find all other contacts that match this one
                for j, other_contact in enumerate(results[i+1:], i+1):
                    if j in processed:
                        continue
                    
                    if self._are_duplicates(contact, other_contact):
                        current_group.append(other_contact)
                        processed.add(j)
                
                groups.append(current_group)
            
            return groups
            
        except Exception as e:
            logger.debug(f"⚠️ Grouping duplicates failed: {e}")
            return [[contact] for contact in results]
    
    def _are_duplicates(self, contact1: ContactResult, contact2: ContactResult) -> bool:
        """
        Determine if two contacts are duplicates using multiple criteria
        """
        try:
            # Exact email match (highest priority)
            if contact1.email and contact2.email:
                if contact1.email.lower() == contact2.email.lower():
                    return True
            
            # Exact LinkedIn URL match
            if contact1.linkedin_url and contact2.linkedin_url:
                if self._normalize_linkedin_url(contact1.linkedin_url) == self._normalize_linkedin_url(contact2.linkedin_url):
                    return True
            
            # Exact phone match
            if contact1.phone and contact2.phone:
                if self._normalize_phone(contact1.phone) == self._normalize_phone(contact2.phone):
                    return True
            
            # Name + Company similarity
            name_similarity = self._calculate_name_similarity(contact1.name, contact2.name)
            company_similarity = self._calculate_company_similarity(contact1.company, contact2.company)
            
            # High name similarity + same company = duplicate
            if (name_similarity >= self.dedup_thresholds["name_similarity"] and 
                company_similarity >= self.dedup_thresholds["company_similarity"]):
                return True
            
            # Very high name similarity + similar company = duplicate
            if (name_similarity >= 0.95 and 
                company_similarity >= self.dedup_thresholds["company_similarity"] * 0.8):
                return True
            
            # Same email domain + high name similarity = likely duplicate
            if contact1.email and contact2.email:
                domain1 = contact1.email.split('@')[1].lower() if '@' in contact1.email else ''
                domain2 = contact2.email.split('@')[1].lower() if '@' in contact2.email else ''
                
                if (domain1 == domain2 and domain1 and 
                    name_similarity >= self.dedup_thresholds["email_domain_name_threshold"]):
                    return True
            
            # Same name + similar position + similar location = duplicate
            if (name_similarity >= self.dedup_thresholds["name_similarity"] and 
                self._calculate_position_similarity(contact1.position, contact2.position) >= self.dedup_thresholds["position_similarity"] and
                self._calculate_location_similarity(contact1.location, contact2.location) >= self.dedup_thresholds["location_similarity"]):
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"⚠️ Duplicate check failed: {e}")
            return False
    
    def _calculate_name_similarity(self, name1: Optional[str], name2: Optional[str]) -> float:
        """Calculate similarity between two names"""
        try:
            if not name1 or not name2:
                return 0.0
            
            # Normalize names
            name1_clean = self._normalize_name(name1)
            name2_clean = self._normalize_name(name2)
            
            if not name1_clean or not name2_clean:
                return 0.0
            
            # Exact match
            if name1_clean == name2_clean:
                return 1.0
            
            # Split into parts and compare
            parts1 = name1_clean.split()
            parts2 = name2_clean.split()
            
            # Check if one name is contained in the other
            if len(parts1) >= 2 and len(parts2) >= 2:
                # Compare first and last names
                first_similarity = SequenceMatcher(None, parts1[0], parts2[0]).ratio()
                last_similarity = SequenceMatcher(None, parts1[-1], parts2[-1]).ratio()
                
                # If first and last names are very similar, consider it a match
                if first_similarity >= 0.9 and last_similarity >= 0.9:
                    return 0.95
                
                # Check for initials match (e.g., "John Smith" vs "J. Smith")
                if (len(parts1[0]) == 1 and parts1[0].lower() == parts2[0][0].lower() and last_similarity >= 0.9):
                    return 0.85
                if (len(parts2[0]) == 1 and parts2[0].lower() == parts1[0][0].lower() and last_similarity >= 0.9):
                    return 0.85
            
            # Overall string similarity
            return SequenceMatcher(None, name1_clean, name2_clean).ratio()
            
        except Exception as e:
            logger.debug(f"⚠️ Name similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_company_similarity(self, company1: Optional[str], company2: Optional[str]) -> float:
        """Calculate similarity between two companies"""
        try:
            if not company1 or not company2:
                return 0.0
            
            # Normalize company names
            comp1_clean = self._normalize_company_name(company1)
            comp2_clean = self._normalize_company_name(company2)
            
            if not comp1_clean or not comp2_clean:
                return 0.0
            
            # Exact match
            if comp1_clean == comp2_clean:
                return 1.0
            
            # Check if one is contained in the other
            if comp1_clean in comp2_clean or comp2_clean in comp1_clean:
                return 0.9
            
            # String similarity
            return SequenceMatcher(None, comp1_clean, comp2_clean).ratio()
            
        except Exception as e:
            logger.debug(f"⚠️ Company similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_position_similarity(self, pos1: Optional[str], pos2: Optional[str]) -> float:
        """Calculate similarity between two positions"""
        try:
            if not pos1 or not pos2:
                return 0.0
            
            pos1_clean = self._normalize_position(pos1)
            pos2_clean = self._normalize_position(pos2)
            
            if not pos1_clean or not pos2_clean:
                return 0.0
            
            if pos1_clean == pos2_clean:
                return 1.0
            
            # Check for common title patterns
            title_synonyms = {
                'engineer': ['developer', 'programmer', 'coder'],
                'manager': ['lead', 'supervisor', 'director'],
                'analyst': ['specialist', 'consultant'],
                'executive': ['director', 'vp', 'president']
            }
            
            for key, synonyms in title_synonyms.items():
                if key in pos1_clean and any(syn in pos2_clean for syn in synonyms):
                    return 0.8
                if key in pos2_clean and any(syn in pos1_clean for syn in synonyms):
                    return 0.8
            
            return SequenceMatcher(None, pos1_clean, pos2_clean).ratio()
            
        except Exception as e:
            logger.debug(f"⚠️ Position similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_location_similarity(self, loc1: Optional[str], loc2: Optional[str]) -> float:
        """Calculate similarity between two locations"""
        try:
            if not loc1 or not loc2:
                return 0.0
            
            loc1_clean = self._normalize_location(loc1)
            loc2_clean = self._normalize_location(loc2)
            
            if not loc1_clean or not loc2_clean:
                return 0.0
            
            if loc1_clean == loc2_clean:
                return 1.0
            
            # Check if locations share city or country
            parts1 = loc1_clean.split(',')
            parts2 = loc2_clean.split(',')
            
            # Compare individual parts
            max_similarity = 0.0
            for part1 in parts1:
                for part2 in parts2:
                    similarity = SequenceMatcher(None, part1.strip(), part2.strip()).ratio()
                    max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            logger.debug(f"⚠️ Location similarity calculation failed: {e}")
            return 0.0
    
    def _merge_contacts(self, contacts: List[ContactResult]) -> ContactResult:
        """
        Merge multiple contact records into a single, comprehensive record
        """
        try:
            if len(contacts) == 1:
                return contacts[0]
            
            # Choose the contact with highest confidence as base
            base_contact = max(contacts, key=lambda c: c.confidence_score)
            
            # Merge data from all contacts
            merged_data = {
                'name': self._merge_field([c.name for c in contacts], 'name'),
                'position': self._merge_field([c.position for c in contacts], 'position'),
                'company': self._merge_field([c.company for c in contacts], 'company'),
                'location': self._merge_field([c.location for c in contacts], 'location'),
                'email': self._merge_field([c.email for c in contacts], 'email'),
                'phone': self._merge_field([c.phone for c in contacts], 'phone'),
                'linkedin_url': self._merge_field([c.linkedin_url for c in contacts], 'linkedin_url'),
                'profile_url': self._merge_field([c.profile_url for c in contacts], 'profile_url'),
                'industry': self._merge_field([c.industry for c in contacts], 'industry'),
                'experience': self._merge_field([c.experience for c in contacts], 'experience'),
                'summary': self._merge_summaries([c.summary for c in contacts]),
                'source': self._merge_sources([c.source for c in contacts]),
                'scraped_at': base_contact.scraped_at,
                'confidence_score': self._calculate_merged_confidence(contacts)
            }
            
            # Create merged contact
            merged_contact = ContactResult(**merged_data)
            
            logger.debug(f"✅ Merged {len(contacts)} contacts for {merged_contact.name}")
            return merged_contact
            
        except Exception as e:
            logger.debug(f"⚠️ Contact merging failed: {e}")
            # Return the contact with highest confidence
            return max(contacts, key=lambda c: c.confidence_score)
    
    def _merge_field(self, values: List[Optional[str]], field_type: str) -> Optional[str]:
        """Merge field values using field-specific logic"""
        try:
            # Remove None and empty values
            valid_values = [v for v in values if v and v.strip()]
            
            if not valid_values:
                return None
            
            if len(valid_values) == 1:
                return valid_values[0]
            
            # Field-specific merging logic
            if field_type == 'name':
                return self._merge_names(valid_values)
            elif field_type == 'email':
                return self._merge_emails(valid_values)
            elif field_type == 'phone':
                return self._merge_phones(valid_values)
            elif field_type in ['linkedin_url', 'profile_url']:
                return self._merge_urls(valid_values)
            elif field_type == 'position':
                return self._merge_positions(valid_values)
            elif field_type == 'company':
                return self._merge_companies(valid_values)
            elif field_type == 'location':
                return self._merge_locations(valid_values)
            else:
                # Default: return the longest/most complete value
                return max(valid_values, key=len)
                
        except Exception as e:
            logger.debug(f"⚠️ Field merge failed for {field_type}: {e}")
            return valid_values[0] if valid_values else None
    
    def _merge_names(self, names: List[str]) -> str:
        """Merge multiple name variations"""
        try:
            # Prefer the most complete name (most words)
            return max(names, key=lambda n: len(n.split()))
        except:
            return names[0]
    
    def _merge_emails(self, emails: List[str]) -> str:
        """Merge multiple emails"""
        try:
            # Prefer business emails over personal ones
            business_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
            
            business_emails = [e for e in emails if not any(domain in e.lower() for domain in business_domains)]
            if business_emails:
                return business_emails[0]
            
            return emails[0]
        except:
            return emails[0]
    
    def _merge_phones(self, phones: List[str]) -> str:
        """Merge multiple phone numbers"""
        try:
            # Prefer the most complete phone number
            return max(phones, key=lambda p: len(re.sub(r'[^\d]', '', p)))
        except:
            return phones[0]
    
    def _merge_urls(self, urls: List[str]) -> str:
        """Merge multiple URLs"""
        try:
            # Prefer LinkedIn URLs, then others
            linkedin_urls = [u for u in urls if 'linkedin.com' in u.lower()]
            if linkedin_urls:
                return linkedin_urls[0]
            return urls[0]
        except:
            return urls[0]
    
    def _merge_positions(self, positions: List[str]) -> str:
        """Merge multiple position titles"""
        try:
            # Prefer the most descriptive/longest title
            return max(positions, key=len)
        except:
            return positions[0]
    
    def _merge_companies(self, companies: List[str]) -> str:
        """Merge multiple company names"""
        try:
            # Prefer the most complete company name
            return max(companies, key=len)
        except:
            return companies[0]
    
    def _merge_locations(self, locations: List[str]) -> str:
        """Merge multiple locations"""
        try:
            # Prefer the most specific location (city, state/country)
            return max(locations, key=lambda l: l.count(',') + len(l))
        except:
            return locations[0]
    
    def _merge_summaries(self, summaries: List[Optional[str]]) -> Optional[str]:
        """Merge multiple summaries"""
        try:
            valid_summaries = [s for s in summaries if s and s.strip()]
            if not valid_summaries:
                return None
            
            # Combine unique parts of summaries
            combined_parts = []
            for summary in valid_summaries:
                if summary not in combined_parts:
                    combined_parts.append(summary)
            
            # Limit combined length
            combined = " | ".join(combined_parts)
            return combined[:500] + "..." if len(combined) > 500 else combined
            
        except:
            return summaries[0] if summaries else None
    
    def _merge_sources(self, sources: List[Optional[str]]) -> Optional[str]:
        """Merge multiple sources"""
        try:
            valid_sources = [s for s in sources if s and s.strip()]
            if not valid_sources:
                return None
            
            unique_sources = list(set(valid_sources))
            return ", ".join(unique_sources)
            
        except:
            return sources[0] if sources else None
    
    def _calculate_merged_confidence(self, contacts: List[ContactResult]) -> float:
        """Calculate confidence for merged contact"""
        try:
            if not contacts:
                return 0.0
            
            # Base confidence is the highest individual confidence
            max_confidence = max(c.confidence_score for c in contacts)
            
            # Bonus for having multiple sources
            source_bonus = min(0.1 * (len(contacts) - 1), 0.2)
            
            # Bonus for having complete information
            merged_contact_data = {
                'email': any(c.email for c in contacts),
                'phone': any(c.phone for c in contacts),
                'linkedin': any(c.linkedin_url for c in contacts),
                'company': any(c.company for c in contacts),
                'position': any(c.position for c in contacts)
            }
            
            completeness_bonus = sum(0.02 for v in merged_contact_data.values() if v)
            
            final_confidence = min(max_confidence + source_bonus + completeness_bonus, 1.0)
            return final_confidence
            
        except:
            return max(c.confidence_score for c in contacts) if contacts else 0.0
    
    def _final_dedup_validation(self, contacts: List[ContactResult]) -> List[ContactResult]:
        """Final validation to catch any remaining duplicates"""
        try:
            # Quick check for exact matches that might have been missed
            seen_emails = set()
            seen_linkedin = set()
            final_contacts = []
            
            for contact in contacts:
                skip = False
                
                # Check email
                if contact.email:
                    email_normalized = contact.email.lower().strip()
                    if email_normalized in seen_emails:
                        skip = True
                    else:
                        seen_emails.add(email_normalized)
                
                # Check LinkedIn URL
                if contact.linkedin_url and not skip:
                    linkedin_normalized = self._normalize_linkedin_url(contact.linkedin_url)
                    if linkedin_normalized in seen_linkedin:
                        skip = True
                    else:
                        seen_linkedin.add(linkedin_normalized)
                
                if not skip:
                    final_contacts.append(contact)
            
            return final_contacts
            
        except Exception as e:
            logger.debug(f"⚠️ Final validation failed: {e}")
            return contacts
    
    def _basic_deduplicate(self, results: List[ContactResult]) -> List[ContactResult]:
        """Fallback basic deduplication method"""
        try:
            seen = set()
            unique_results = []
            
            for result in results:
                # Create identifier for deduplication
                identifier_parts = []
                
                if result.email:
                    identifier_parts.append(result.email.lower())
                else:
                    if result.name:
                        identifier_parts.append(result.name.lower())
                    if result.company:
                        identifier_parts.append(result.company.lower())
                
                identifier = "|".join(identifier_parts)
                
                if identifier and identifier not in seen:
                    seen.add(identifier)
                    unique_results.append(result)
            
            return unique_results
            
        except Exception as e:
            logger.debug(f"⚠️ Basic deduplication failed: {e}")
            return results
    
    # Normalization helper methods
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison"""
        try:
            if not name:
                return ""
            
            # Remove extra whitespace, convert to lowercase
            normalized = re.sub(r'\s+', ' ', name.strip().lower())
            
            # Remove common prefixes/suffixes
            prefixes = ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.']
            suffixes = ['jr.', 'sr.', 'ii', 'iii', 'iv']
            
            words = normalized.split()
            words = [w for w in words if w not in prefixes and w not in suffixes]
            
            return ' '.join(words)
            
        except:
            return name.lower() if name else ""
    
    def _normalize_company_name(self, company: str) -> str:
        """Normalize company name for comparison"""
        try:
            if not company:
                return ""
            
            normalized = company.lower().strip()
            
            # Remove common company suffixes
            suffixes = ['inc.', 'inc', 'corp.', 'corp', 'ltd.', 'ltd', 'llc', 'llp', 
                       'co.', 'co', 'company', 'corporation', 'limited']
            
            for suffix in suffixes:
                if normalized.endswith(' ' + suffix):
                    normalized = normalized[:-len(suffix)-1].strip()
                elif normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)].strip()
            
            # Remove extra whitespace
            normalized = re.sub(r'\s+', ' ', normalized)
            
            return normalized
            
        except:
            return company.lower() if company else ""
    
    def _normalize_position(self, position: str) -> str:
        """Normalize position title for comparison"""
        try:
            if not position:
                return ""
            
            normalized = position.lower().strip()
            
            # Remove common words that don't affect meaning
            noise_words = ['senior', 'junior', 'lead', 'principal', 'chief', 'head of']
            
            for word in noise_words:
                normalized = normalized.replace(word, '').strip()
            
            # Normalize common abbreviations
            normalized = normalized.replace('mgr', 'manager')
            normalized = normalized.replace('eng', 'engineer')
            normalized = normalized.replace('dev', 'developer')
            
            # Remove extra whitespace
            normalized = re.sub(r'\s+', ' ', normalized)
            
            return normalized
            
        except:
            return position.lower() if position else ""
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location for comparison"""
        try:
            if not location:
                return ""
            
            normalized = location.lower().strip()
            
            # Remove common abbreviations and standardize
            normalized = normalized.replace('usa', 'united states')
            normalized = normalized.replace('uk', 'united kingdom')
            normalized = normalized.replace('ca', 'california')
            normalized = normalized.replace('ny', 'new york')
            
            # Remove extra whitespace
            normalized = re.sub(r'\s+', ' ', normalized)
            
            return normalized
            
        except:
            return location.lower() if location else ""
    
    def _normalize_linkedin_url(self, url: str) -> str:
        """Normalize LinkedIn URL for comparison"""
        try:
            if not url:
                return ""
            
            # Extract the profile ID from LinkedIn URL
            match = re.search(r'/in/([^/?]+)', url.lower())
            return match.group(1) if match else url.lower()
            
        except:
            return url.lower() if url else ""
    
    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number for comparison"""
        try:
            if not phone:
                return ""
            
            # Remove all non-digit characters
            digits_only = re.sub(r'[^\d]', '', phone)
            
            # Handle US numbers (remove country code if present)
            if len(digits_only) == 11 and digits_only.startswith('1'):
                digits_only = digits_only[1:]
            
            return digits_only
            
        except:
            return phone if phone else ""
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.browser_page:
                self.browser_page.quit()
                self.browser_page = None
                self._browser_created = False
                logger.info("✅ Browser cleaned up")
        except Exception as e:
            logger.debug(f"⚠️ Cleanup error: {e}")

# Pydantic models for API
class SearchRequest(BaseModel):
    industry: Optional[str] = Field(None, description="Target industry")
    position: Optional[str] = Field(None, description="Job position/title")
    company: Optional[str] = Field(None, description="Company name")
    country: Optional[str] = Field(None, description="Country")
    city: Optional[str] = Field(None, description="City")
    keywords: Optional[str] = Field(None, description="Additional keywords")
    experience_level: Optional[str] = Field(None, description="Experience level")
    company_size: Optional[str] = Field(None, description="Company size")
    max_results: int = Field(50, ge=1, le=200, description="Maximum results")
    enable_deduplication: bool = Field(True, description="Enable advanced deduplication")
    dedup_strictness: str = Field("medium", description="Deduplication strictness: loose, medium, strict")

class ContactResponse(BaseModel):
    name: str
    position: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    profile_url: Optional[str] = None
    industry: Optional[str] = None
    experience: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = None
    scraped_at: Optional[str] = None
    confidence_score: float

class DeduplicationRequest(BaseModel):
    contacts: List[ContactResponse] = Field(..., description="List of contacts to deduplicate")
    strictness: str = Field("medium", description="Deduplication strictness: loose, medium, strict")

class DeduplicationResponse(BaseModel):
    success: bool
    message: str
    original_count: int
    deduplicated_count: int
    duplicates_removed: int
    contacts: List[ContactResponse]
    dedup_stats: dict

class SearchResponse(BaseModel):
    success: bool
    message: str
    total_results: int
    contacts: List[ContactResponse]
    search_params: dict

# FastAPI Application
app = FastAPI(
    title="Contact & People Scraper API",
    description="Professional contact scraping service with headless browsing",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:8080",  # Local frontend
        "https://*.vercel.app",   # All Vercel deployments
        "https://contact-scraper-frontend.vercel.app",  # Your specific Vercel URL (update after deployment)
        os.getenv("FRONTEND_URL", ""),  # Environment variable
        "*"  # Keep this for development, remove in production if needed
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global scraper instance with deduplication
scraper = ContactScraper(headless=True, stealth=True, dedup_strictness="medium")

@app.on_event("startup")
async def startup_event():
    """Initialize scraper on startup"""
    logger.info("🚀 Contact Scraper API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    await scraper.close()
    logger.info("👋 Contact Scraper API shutting down...")

@app.get("/")
async def root():
    """Railway health check endpoint"""
    return {
        "message": "Contact & People Scraper API",
        "status": "healthy",
        "environment": RAILWAY_ENV,
        "version": "1.0.0"
    }
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "scraper_ready": scraper._browser_created
    }

@app.post("/search", response_model=SearchResponse)
async def search_contacts(request: SearchRequest):
    """
    Search for contacts based on provided parameters
    """
    try:
        # Convert request to search parameters
        search_params = SearchParameters(
            industry=request.industry,
            position=request.position,
            company=request.company,
            country=request.country,
            city=request.city,
            keywords=request.keywords,
            experience_level=request.experience_level,
            company_size=request.company_size,
            max_results=request.max_results
        )
        
        # Perform search
        results = await scraper.search_contacts(search_params, request.enable_deduplication)
        
        # Convert results to response format
        contact_responses = [
            ContactResponse(**asdict(contact)) for contact in results
        ]
        
        return SearchResponse(
            success=True,
            message=f"Found {len(results)} contacts (deduplication: {'enabled' if request.enable_deduplication else 'disabled'})",
            total_results=len(results),
            contacts=contact_responses,
            search_params=asdict(search_params)
        )
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/linkedin", response_model=SearchResponse)
async def search_linkedin_only(request: SearchRequest):
    """
    Search for contacts specifically on LinkedIn
    WARNING: Ensure compliance with LinkedIn Terms of Service
    Consider using LinkedIn's official API for production use
    """
    try:
        logger.warning("⚠️ LinkedIn-only search requested - ensure ToS compliance!")
        
        # Convert request to search parameters
        search_params = SearchParameters(
            industry=request.industry,
            position=request.position,
            company=request.company,
            country=request.country,
            city=request.city,
            keywords=request.keywords,
            experience_level=request.experience_level,
            company_size=request.company_size,
            max_results=min(request.max_results, 20)  # Lower limit for LinkedIn
        )
        
        # Perform LinkedIn-specific search
        results = await scraper._search_linkedin_carefully(search_params)
        
        # Convert results to response format
        contact_responses = [
            ContactResponse(**asdict(contact)) for contact in results
        ]
        
        return SearchResponse(
            success=True,
            message=f"Found {len(results)} LinkedIn contacts (ToS compliant search)",
            total_results=len(results),
            contacts=contact_responses,
            search_params=asdict(search_params)
        )
        
    except Exception as e:
        logger.error(f"❌ LinkedIn search failed: {e}")
        raise HTTPException(status_code=500, detail=f"LinkedIn search failed: {str(e)}")

@app.get("/legal/disclaimer")
async def legal_disclaimer():
    """
    Legal disclaimer and Terms of Service information
    """
    return {
        "disclaimer": "IMPORTANT LEGAL NOTICE",
        "message": "This scraping tool must be used in compliance with all applicable laws and website Terms of Service",
        "linkedin_warning": {
            "notice": "LinkedIn has strict Terms of Service regarding automated data collection",
            "recommendation": "Use LinkedIn's official API for production applications",
            "api_url": "https://developer.linkedin.com/",
            "compliance": "Ensure you have proper authorization and respect rate limits"
        },
        "general_guidelines": [
            "Only collect publicly available information",
            "Respect robots.txt files and rate limits",
            "Do not overwhelm target servers",
            "Comply with GDPR and privacy regulations",
            "Obtain proper consent for data collection",
            "Use data only for legitimate business purposes"
        ],
        "user_responsibility": "Users are solely responsible for ensuring their use of this tool complies with all applicable laws and website terms of service"
    }

@app.get("/linkedin/status")
async def linkedin_scraping_status():
    """
    Get current LinkedIn scraping status and recommendations
    """
    return {
        "current_requests": scraper.linkedin_request_count,
        "hourly_limit": scraper.linkedin_max_requests_per_hour,
        "remaining_requests": max(0, scraper.linkedin_max_requests_per_hour - scraper.linkedin_request_count),
        "rate_limit_delay": scraper.linkedin_delay,
        "recommendation": "Consider using LinkedIn's official API for higher volume and more reliable access",
        "api_alternatives": {
            "linkedin_api": "https://developer.linkedin.com/",
            "sales_navigator": "https://business.linkedin.com/sales-solutions/sales-navigator",
            "recruiter": "https://business.linkedin.com/talent-solutions/recruiter"
        },
        "status": "operational" if scraper.linkedin_request_count < scraper.linkedin_max_requests_per_hour else "rate_limited"
    }

@app.post("/deduplicate", response_model=DeduplicationResponse)
async def deduplicate_contacts(request: DeduplicationRequest):
    """
    Deduplicate a list of contacts using advanced algorithms
    """
    try:
        logger.info(f"🔄 Deduplication requested for {len(request.contacts)} contacts with {request.strictness} strictness")
        
        # Convert ContactResponse objects to ContactResult objects
        contact_results = []
        for contact_data in request.contacts:
            contact_dict = contact_data.dict()
            contact_result = ContactResult(**contact_dict)
            contact_results.append(contact_result)
        
        # Perform deduplication
        deduplicated_contacts, stats = await scraper.deduplicate_contacts(
            contact_results, 
            request.strictness
        )
        
        # Convert back to response format
        deduplicated_responses = [
            ContactResponse(**asdict(contact)) for contact in deduplicated_contacts
        ]
        
        return DeduplicationResponse(
            success=True,
            message=f"Deduplication complete. Removed {stats.get('duplicates_removed', 0)} duplicates",
            original_count=stats.get('original_count', 0),
            deduplicated_count=stats.get('deduplicated_count', 0),
            duplicates_removed=stats.get('duplicates_removed', 0),
            contacts=deduplicated_responses,
            dedup_stats=stats
        )
        
    except Exception as e:
        logger.error(f"❌ Deduplication failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deduplication failed: {str(e)}")

@app.get("/deduplicate/config")
async def get_deduplication_config():
    """
    Get current deduplication configuration and available options
    """
    return {
        "current_strictness": scraper.dedup_strictness,
        "current_thresholds": scraper.dedup_thresholds,
        "available_strictness_levels": {
            "loose": {
                "description": "More permissive - catches obvious duplicates only",
                "use_case": "When you want to keep more contacts and remove only clear duplicates",
                "thresholds": scraper._get_dedup_thresholds("loose")
            },
            "medium": {
                "description": "Balanced approach - good for most use cases",
                "use_case": "Default setting - balances duplicate detection with false positives",
                "thresholds": scraper._get_dedup_thresholds("medium")
            },
            "strict": {
                "description": "Very restrictive - only keeps contacts that are clearly different",
                "use_case": "When you want maximum deduplication, even if some unique contacts are removed",
                "thresholds": scraper._get_dedup_thresholds("strict")
            }
        },
        "deduplication_criteria": [
            "Exact email match (highest priority)",
            "Exact LinkedIn URL match",
            "Exact phone number match",
            "Name + Company similarity",
            "Email domain + Name similarity",
            "Name + Position + Location similarity"
        ]
    }

@app.post("/deduplicate/config")
async def update_deduplication_config(strictness: str):
    """
    Update deduplication strictness level
    """
    valid_levels = ["loose", "medium", "strict"]
    
    if strictness not in valid_levels:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid strictness level. Must be one of: {valid_levels}"
        )
    
    old_strictness = scraper.dedup_strictness
    scraper.set_dedup_strictness(strictness)
    
    return {
        "success": True,
        "message": f"Deduplication strictness updated from {old_strictness} to {strictness}",
        "old_strictness": old_strictness,
        "new_strictness": strictness,
        "new_thresholds": scraper.dedup_thresholds
    }

@app.get("/deduplicate/test")
async def test_deduplication():
    """
    Test deduplication with sample data
    """
    # Create sample duplicate contacts for testing
    sample_contacts = [
        ContactResult(
            name="John Smith",
            position="Software Engineer",
            company="Tech Corp",
            email="john.smith@techcorp.com",
            location="San Francisco, CA",
            source="Test Data",
            confidence_score=0.9
        ),
        ContactResult(
            name="J. Smith",
            position="Senior Software Engineer", 
            company="Tech Corp Inc.",
            email="j.smith@techcorp.com",
            location="San Francisco, California",
            source="Test Data",
            confidence_score=0.8
        ),
        ContactResult(
            name="John D. Smith",
            position="Software Developer",
            company="Tech Corporation",
            linkedin_url="https://linkedin.com/in/johnsmith123",
            location="SF, CA",
            source="Test Data",
            confidence_score=0.85
        ),
        ContactResult(
            name="Jane Doe",
            position="Product Manager",
            company="StartupXYZ",
            email="jane@startupxyz.com",
            source="Test Data",
            confidence_score=0.7
        ),
        ContactResult(
            name="Jane R. Doe",
            position="Senior Product Manager",
            company="StartupXYZ",
            email="jane.doe@startupxyz.com",
            location="New York, NY",
            source="Test Data",
            confidence_score=0.8
        )
    ]
    
    # Test with different strictness levels
    results = {}
    
    for strictness in ["loose", "medium", "strict"]:
        deduplicated, stats = await scraper.deduplicate_contacts(sample_contacts, strictness)
        results[strictness] = {
            "original_count": len(sample_contacts),
            "deduplicated_count": len(deduplicated),
            "duplicates_removed": len(sample_contacts) - len(deduplicated),
            "duplicate_rate": round(((len(sample_contacts) - len(deduplicated)) / len(sample_contacts)) * 100, 2),
            "contacts": [asdict(contact) for contact in deduplicated]
        }
    
    return {
        "test_description": "Sample deduplication test with 5 contacts containing 2 duplicate groups",
        "sample_contacts": [asdict(contact) for contact in sample_contacts],
        "results_by_strictness": results,
        "recommendations": {
            "loose": "Use when you want to keep maximum contacts",
            "medium": "Recommended for most use cases",
            "strict": "Use when duplicate removal is critical"
        }
    }

@app.post("/search/async")
async def search_contacts_async(request: SearchRequest, background_tasks: BackgroundTasks):
    """
    Start asynchronous contact search (for long-running searches)
    """
    # Generate search ID
    search_id = hashlib.md5(f"{datetime.now().isoformat()}{request}".encode()).hexdigest()[:8]
    
    # Start background task
    background_tasks.add_task(perform_background_search, search_id, request)
    
    return {
        "search_id": search_id,
        "status": "started",
        "message": "Search started in background",
        "check_url": f"/search/status/{search_id}"
    }

# Background search storage (in production, use Redis or database)
search_results = {}

async def perform_background_search(search_id: str, request: SearchRequest):
    """Perform search in background"""
    try:
        search_results[search_id] = {"status": "running", "progress": 0}
        
        search_params = SearchParameters(
            industry=request.industry,
            position=request.position,
            company=request.company,
            country=request.country,
            city=request.city,
            keywords=request.keywords,
            experience_level=request.experience_level,
            company_size=request.company_size,
            max_results=request.max_results
        )
        
        results = await scraper.search_contacts(search_params)
        
        search_results[search_id] = {
            "status": "completed",
            "progress": 100,
            "results": [asdict(contact) for contact in results],
            "total_results": len(results)
        }
        
    except Exception as e:
        search_results[search_id] = {
            "status": "failed",
            "error": str(e)
        }

@app.get("/search/status/{search_id}")
async def get_search_status(search_id: str):
    """Get status of background search"""
    if search_id not in search_results:
        raise HTTPException(status_code=404, detail="Search ID not found")
    
    return search_results[search_id]

@app.get("/api/stats")
async def get_api_stats():
    """Get API usage statistics"""
    return {
        "linkedin_requests": scraper.linkedin_request_count,
        "linkedin_limit": scraper.linkedin_max_requests_per_hour,
        "browser_status": "active" if scraper._browser_created else "inactive",
        "dedup_strictness": scraper.dedup_strictness,
        "background_searches": len(search_results),
        "uptime": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/clear-cache")
async def clear_cache():
    """Clear background search cache"""
    global search_results
    old_count = len(search_results)
    search_results = {}
    
    return {
        "success": True,
        "message": f"Cleared {old_count} cached search results",
        "remaining_results": len(search_results)
    }

@app.post("/api/test-browser")
async def test_browser():
    """Test browser functionality"""
    try:
        success = await scraper._create_browser()
        
        if success:
            # Test basic functionality
            scraper.browser_page.get("https://httpbin.org/user-agent")
            user_agent = scraper.browser_page.html
            
            return {
                "success": True,
                "message": "Browser test successful",
                "browser_created": scraper._browser_created,
                "test_url": "https://httpbin.org/user-agent",
                "response_length": len(user_agent) if user_agent else 0
            }
        else:
            return {
                "success": False,
                "message": "Browser creation failed",
                "browser_created": False
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Browser test failed: {str(e)}",
            "browser_created": scraper._browser_created
        }

@app.get("/api/browser/restart")
async def restart_browser():
    """Restart browser instance"""
    try:
        # Close existing browser
        await scraper.close()
        
        # Create new browser
        success = await scraper._create_browser()
        
        return {
            "success": success,
            "message": "Browser restarted successfully" if success else "Browser restart failed",
            "browser_status": "active" if success else "inactive"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Browser restart failed: {str(e)}",
            "browser_status": "error"
        }

# Advanced search endpoints
@app.post("/search/advanced")
async def advanced_search(request: SearchRequest):
    """
    Advanced search with enhanced filtering and processing
    """
    try:
        logger.info(f"🔍 Advanced search requested: {request}")
        
        # Convert request to search parameters
        search_params = SearchParameters(
            industry=request.industry,
            position=request.position,
            company=request.company,
            country=request.country,
            city=request.city,
            keywords=request.keywords,
            experience_level=request.experience_level,
            company_size=request.company_size,
            max_results=request.max_results
        )
        
        # Perform search with enhanced processing
        results = await scraper.search_contacts(search_params, request.enable_deduplication)
        
        # Enhanced result processing
        enhanced_results = []
        for contact in results:
            # Add email pattern generation
            if contact.name and contact.company and not contact.email:
                email_patterns = scraper._generate_email_patterns(contact.name, contact.company)
                contact.summary = f"Suggested emails: {', '.join(email_patterns[:3])}"
            
            enhanced_results.append(contact)
        
        # Generate search analytics
        analytics = {
            "source_breakdown": {},
            "confidence_stats": {
                "high": len([c for c in enhanced_results if c.confidence_score >= 0.8]),
                "medium": len([c for c in enhanced_results if 0.5 <= c.confidence_score < 0.8]),
                "low": len([c for c in enhanced_results if c.confidence_score < 0.5])
            },
            "data_completeness": {
                "with_email": len([c for c in enhanced_results if c.email]),
                "with_phone": len([c for c in enhanced_results if c.phone]),
                "with_linkedin": len([c for c in enhanced_results if c.linkedin_url]),
                "with_location": len([c for c in enhanced_results if c.location])
            }
        }
        
        # Count by source
        for contact in enhanced_results:
            source = contact.source or "Unknown"
            analytics["source_breakdown"][source] = analytics["source_breakdown"].get(source, 0) + 1
        
        # Convert results to response format
        contact_responses = [
            ContactResponse(**asdict(contact)) for contact in enhanced_results
        ]
        
        return {
            "success": True,
            "message": f"Advanced search completed. Found {len(enhanced_results)} contacts",
            "total_results": len(enhanced_results),
            "contacts": contact_responses,
            "search_params": asdict(search_params),
            "analytics": analytics,
            "search_quality": {
                "avg_confidence": sum(c.confidence_score for c in enhanced_results) / len(enhanced_results) if enhanced_results else 0,
                "data_richness": sum(1 for c in enhanced_results if any([c.email, c.phone, c.linkedin_url])) / len(enhanced_results) if enhanced_results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Advanced search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")

# Export endpoints
@app.post("/export/csv")
async def export_contacts_csv(contacts: List[ContactResponse]):
    """
    Export contacts to CSV format
    """
    try:
        import csv
        import io
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            'name', 'position', 'company', 'location', 'email', 'phone',
            'linkedin_url', 'industry', 'source', 'confidence_score'
        ])
        
        writer.writeheader()
        for contact in contacts:
            writer.writerow({
                'name': contact.name,
                'position': contact.position or '',
                'company': contact.company or '',
                'location': contact.location or '',
                'email': contact.email or '',
                'phone': contact.phone or '',
                'linkedin_url': contact.linkedin_url or '',
                'industry': contact.industry or '',
                'source': contact.source or '',
                'confidence_score': contact.confidence_score
            })
        
        csv_content = output.getvalue()
        output.close()
        
        return {
            "success": True,
            "format": "csv",
            "content": csv_content,
            "records_exported": len(contacts),
            "filename": f"contacts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")

@app.post("/export/json")
async def export_contacts_json(contacts: List[ContactResponse]):
    """
    Export contacts to JSON format
    """
    try:
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_contacts": len(contacts),
                "format": "json",
                "version": "1.0.0"
            },
            "contacts": [contact.dict() for contact in contacts]
        }
        
        return {
            "success": True,
            "format": "json",
            "data": export_data,
            "records_exported": len(contacts),
            "filename": f"contacts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON export failed: {str(e)}")

# Validation endpoints
@app.post("/validate/email")
async def validate_email_patterns(contact: ContactResponse):
    """
    Generate and validate email patterns for a contact
    """
    try:
        if not contact.name or not contact.company:
            raise HTTPException(status_code=400, detail="Name and company required for email pattern generation")
        
        # Convert to ContactResult for processing
        contact_result = ContactResult(**contact.dict())
        
        # Generate email patterns
        email_patterns = scraper._generate_email_patterns(contact.name, contact.company)
        
        # Basic email format validation
        import re
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
        valid_patterns = [email for email in email_patterns if re.match(email_regex, email)]
        
        return {
            "success": True,
            "contact_name": contact.name,
            "company": contact.company,
            "generated_patterns": email_patterns,
            "valid_patterns": valid_patterns,
            "pattern_count": len(valid_patterns),
            "confidence": "medium" if len(valid_patterns) > 3 else "low"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email validation failed: {str(e)}")

@app.post("/search/debug")
async def debug_search(request: SearchRequest):
    """
    Debug search to see what's happening step by step
    """
    try:
        # Convert request to search parameters
        search_params = SearchParameters(
            industry=request.industry,
            position=request.position,
            company=request.company,
            country=request.country,
            city=request.city,
            keywords=request.keywords,
            experience_level=request.experience_level,
            company_size=request.company_size,
            max_results=min(request.max_results, 10)  # Limit for debugging
        )
        
        debug_info = {
            "search_params": asdict(search_params),
            "google_query": scraper._build_google_query(search_params),
            "steps": [],
            "results_by_source": {}
        }
        
        # Test Google search specifically
        try:
            if not scraper.browser_page:
                await scraper._create_browser()
                debug_info["steps"].append("✅ Browser created successfully")
            
            # Build and test Google search
            search_query = scraper._build_google_query(search_params)
            google_url = f"https://www.google.com/search?q={quote(search_query)}"
            
            debug_info["steps"].append(f"🔍 Google URL: {google_url}")
            
            scraper.browser_page.get(google_url)
            await asyncio.sleep(3)
            
            # Check for results
            search_results = scraper.browser_page.eles('css:.g')
            debug_info["steps"].append(f"📊 Found {len(search_results)} Google result elements")
            
            # Parse a few results for debugging
            parsed_results = []
            for i, result in enumerate(search_results[:5]):
                try:
                    # Get basic info
                    title_elem = result.ele('css:h3', timeout=1)
                    link_elem = result.ele('css:a', timeout=1)
                    snippet_elem = result.ele('css:.VwiC3b', timeout=1)
                    
                    result_info = {
                        "index": i,
                        "title": title_elem.text if title_elem else "No title",
                        "url": link_elem.attr('href') if link_elem else "No URL",
                        "snippet": snippet_elem.text if snippet_elem else "No snippet"
                    }
                    
                    # Try to parse it
                    contact = scraper._parse_google_result(result, search_params)
                    result_info["parsed_contact"] = asdict(contact) if contact else None
                    result_info["parsing_success"] = contact is not None
                    
                    parsed_results.append(result_info)
                    
                except Exception as e:
                    parsed_results.append({
                        "index": i,
                        "error": str(e)
                    })
            
            debug_info["google_results_sample"] = parsed_results
            debug_info["steps"].append(f"📝 Parsed {len([r for r in parsed_results if r.get('parsing_success')])} contacts from Google")
            
        except Exception as e:
            debug_info["steps"].append(f"❌ Google search failed: {str(e)}")
        
        return {
            "success": True,
            "message": "Debug search completed",
            "debug_info": debug_info,
            "recommendations": [
                "Try adding more specific search terms (position, company, industry)",
                "Check if the Google query looks reasonable",
                "Verify that search results contain contact information",
                "Consider using more targeted searches instead of just location"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Debug search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug search failed: {str(e)}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,  # Disable in production
        log_level="info"
    )

# Example usage and API documentation:
"""
PROFESSIONAL CONTACT & PEOPLE SCRAPER API

🚀 Features:
- Multi-source contact scraping (Google, LinkedIn, Business Directories, Company Websites)
- Advanced deduplication with fuzzy matching
- LinkedIn integration with ToS compliance warnings
- Railway deployment optimized
- Real-time browser automation with stealth features
- Comprehensive API with multiple endpoints

📋 Main Endpoints:
- POST /search - Main contact search
- POST /search/linkedin - LinkedIn-only search (use responsibly)
- POST /search/async - Background search for large requests
- POST /deduplicate - Advanced contact deduplication
- POST /search/advanced - Enhanced search with analytics
- GET /health - Health check
- GET /legal/disclaimer - Legal information

🔧 Advanced Features:
- Rate limiting and stealth browsing
- Email pattern generation
- Confidence scoring
- Source attribution
- Background search processing
- CSV/JSON export capabilities
- Browser management endpoints

⚖️ Legal Compliance:
- LinkedIn ToS warnings and compliance features
- Rate limiting to respect target sites
- Public data only extraction
- User responsibility disclaimers

🚀 Installation:
pip install DrissionPage fastapi uvicorn python-multipart

🔗 Example Usage:
curl -X POST "https://your-railway-app.railway.app/search" \
-H "Content-Type: application/json" \
-d '{
    "position": "Software Engineer",
    "company": "Google",
    "country": "United States",
    "max_results": 10,
    "enable_deduplication": true
}'

📊 Advanced Search:
curl -X POST "https://your-railway-app.railway.app/search/advanced" \
-H "Content-Type: application/json" \
-d '{
    "industry": "technology",
    "position": "data scientist",
    "experience_level": "senior",
    "company_size": "large",
    "max_results": 25
}'

🔄 Deduplication Only:
curl -X POST "https://your-railway-app.railway.app/deduplicate" \
-H "Content-Type: application/json" \
-d '{
    "contacts": [/* your contact array */],
    "strictness": "medium"
}'

⚠️ Important Notes:
1. Always respect website Terms of Service
2. Use LinkedIn's official API for production LinkedIn data
3. Implement proper rate limiting
4. Only collect publicly available information
5. Comply with GDPR and privacy regulations

🌐 Frontend Integration:
The API is CORS-enabled and ready for frontend integration.
All endpoints return structured JSON responses.
"""
        valid_patterns = [email for email in email_patterns if re.match(email_regex, email)]
        
        return {
            "success": True,
            "contact_name": contact.name,
            "company": contact.company,
            "generated_patterns": email_patterns,
            "valid_patterns": valid_patterns,
            "pattern_count": len(valid_patterns),
            "confidence": "medium" if len(valid_patterns) > 3 else "low"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email validation failed: {str(e)}")

@app.post("/search/debug")
async def debug_search(request: SearchRequest):
    """
    Debug search to see what's happening step by step
    """
    try:
        # Convert request to search parameters
        search_params = SearchParameters(
            industry=request.industry,
            position=request.position,
            company=request.company,
            country=request.country,
            city=request.city,
            keywords=request.keywords,
            experience_level=request.experience_level,
            company_size=request.company_size,
            max_results=min(request.max_results, 10)  # Limit for debugging
        )
        
        debug_info = {
            "search_params": asdict(search_params),
            "google_query": scraper._build_google_query(search_params),
            "steps": [],
            "results_by_source": {}
        }
        
        # Test Google search specifically
        try:
            if not scraper.browser_page:
                await scraper._create_browser()
                debug_info["steps"].append("✅ Browser created successfully")
            
            # Build and test Google search
            search_query = scraper._build_google_query(search_params)
            google_url = f"https://www.google.com/search?q={quote(search_query)}"
            
            debug_info["steps"].append(f"🔍 Google URL: {google_url}")
            
            scraper.browser_page.get(google_url)
            await asyncio.sleep(3)
            
            # Check for results
            search_results = scraper.browser_page.eles('css:.g')
            debug_info["steps"].append(f"📊 Found {len(search_results)} Google result elements")
            
            # Parse a few results for debugging
            parsed_results = []
            for i, result in enumerate(search_results[:5]):
                try:
                    # Get basic info
                    title_elem = result.ele('css:h3', timeout=1)
                    link_elem = result.ele('css:a', timeout=1)
                    snippet_elem = result.ele('css:.VwiC3b', timeout=1)
                    
                    result_info = {
                        "index": i,
                        "title": title_elem.text if title_elem else "No title",
                        "url": link_elem.attr('href') if link_elem else "No URL",
                        "snippet": snippet_elem.text if snippet_elem else "No snippet"
                    }
                    
                    # Try to parse it
                    contact = scraper._parse_google_result(result, search_params)
                    result_info["parsed_contact"] = asdict(contact) if contact else None
                    result_info["parsing_success"] = contact is not None
                    
                    parsed_results.append(result_info)
                    
                except Exception as e:
                    parsed_results.append({
                        "index": i,
                        "error": str(e)
                    })
            
            debug_info["google_results_sample"] = parsed_results
            debug_info["steps"].append(f"📝 Parsed {len([r for r in parsed_results if r.get('parsing_success')])} contacts from Google")
            
        except Exception as e:
            debug_info["steps"].append(f"❌ Google search failed: {str(e)}")
        
        return {
            "success": True,
            "message": "Debug search completed",
            "debug_info": debug_info,
            "recommendations": [
                "Try adding more specific search terms (position, company, industry)",
                "Check if the Google query looks reasonable",
                "Verify that search results contain contact information",
                "Consider using more targeted searches instead of just location"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Debug search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug search failed: {str(e)}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,  # Disable in production
        log_level="info"
    )

# Example usage and API documentation:
"""
PROFESSIONAL CONTACT & PEOPLE SCRAPER API

🚀 Features:
- Multi-source contact scraping (Google, LinkedIn, Business Directories, Company Websites)
- Advanced deduplication with fuzzy matching
- LinkedIn integration with ToS compliance warnings
- Railway deployment optimized
- Real-time browser automation with stealth features
- Comprehensive API with multiple endpoints

📋 Main Endpoints:
- POST /search - Main contact search
- POST /search/linkedin - LinkedIn-only search (use responsibly)
- POST /search/async - Background search for large requests
- POST /deduplicate - Advanced contact deduplication
- POST /search/advanced - Enhanced search with analytics
- GET /health - Health check
- GET /legal/disclaimer - Legal information

🔧 Advanced Features:
- Rate limiting and stealth browsing
- Email pattern generation
- Confidence scoring
- Source attribution
- Background search processing
- CSV/JSON export capabilities
- Browser management endpoints

⚖️ Legal Compliance:
- LinkedIn ToS warnings and compliance features
- Rate limiting to respect target sites
- Public data only extraction
- User responsibility disclaimers

🚀 Installation:
pip install DrissionPage fastapi uvicorn python-multipart

🔗 Example Usage:
curl -X POST "https://your-railway-app.railway.app/search" \
-H "Content-Type: application/json" \
-d '{
    "position": "Software Engineer",
    "company": "Google",
    "country": "United States",
    "max_results": 10,
    "enable_deduplication": true
}'

📊 Advanced Search:
curl -X POST "https://your-railway-app.railway.app/search/advanced" \
-H "Content-Type: application/json" \
-d '{
    "industry": "technology",
    "position": "data scientist",
    "experience_level": "senior",
    "company_size": "large",
    "max_results": 25
}'

🔄 Deduplication Only:
curl -X POST "https://your-railway-app.railway.app/deduplicate" \
-H "Content-Type: application/json" \
-d '{
    "contacts": [/* your contact array */],
    "strictness": "medium"
}'

⚠️ Important Notes:
1. Always respect website Terms of Service
2. Use LinkedIn's official API for production LinkedIn data
3. Implement proper rate limiting
4. Only collect publicly available information
5. Comply with GDPR and privacy regulations

🌐 Frontend Integration:
The API is CORS-enabled and ready for frontend integration.
All endpoints return structured JSON responses.
"""
