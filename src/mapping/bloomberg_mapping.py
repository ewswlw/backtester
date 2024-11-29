import yaml
from pathlib import Path
from typing import Dict, List

class BloombergMapping:
    """Class to handle Bloomberg data mappings from configuration."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def get_tickers_for_table(self, table: str) -> List[str]:
        """Get list of tickers for a specific table."""
        securities = self.config.get(table, {}).get('securities', [])
        return [sec['ticker'] for sec in securities]
        
    def get_field_for_table(self, table: str) -> str:
        """Get Bloomberg field for a specific table."""
        return self.config.get(table, {}).get('field', '')
        
    def get_custom_names_mapping(self, table: str) -> Dict[str, str]:
        """Get mapping of Bloomberg tickers to custom names."""
        securities = self.config.get(table, {}).get('securities', [])
        return {sec['ticker']: sec['custom_name'] for sec in securities}
