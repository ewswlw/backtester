"""Configuration validation utilities."""
import yaml
from pathlib import Path
from typing import Dict, Any

def load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the validated configuration
        
    Raises:
        ValueError: If required sections or settings are missing
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['settings', 'sprds', 'derv']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate settings
    required_settings = ['default_start_date', 'data_directory', 'log_directory']
    for setting in required_settings:
        if setting not in config['settings']:
            raise ValueError(f"Missing required setting: {setting}")
    
    # Validate data sections
    for section in ['sprds', 'derv']:
        if 'field' not in config[section]:
            raise ValueError(f"Missing 'field' in {section} section")
        if 'securities' not in config[section]:
            raise ValueError(f"Missing 'securities' in {section} section")
        if not isinstance(config[section]['securities'], list):
            raise ValueError(f"'securities' in {section} section must be a list")
        for security in config[section]['securities']:
            if 'ticker' not in security:
                raise ValueError(f"Missing 'ticker' in security of {section} section")
            if 'custom_name' not in security:
                raise ValueError(f"Missing 'custom_name' in security of {section} section")
    
    # Print configuration summary
    print("Configuration Summary:")
    print("\nSettings:")
    for key, value in config['settings'].items():
        print(f"{key}: {value}")
    
    print("\nData Categories:")
    for category in ['sprds', 'derv']:
        num_securities = len(config[category]['securities'])
        field = config[category]['field']
        print(f"{category}:")
        print(f"  Field: {field}")
        print(f"  Securities: {num_securities}")
        for security in config[category]['securities']:
            print(f"    - {security['custom_name']}: {security['ticker']}")
    
    return config

def get_tickers_by_category(config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Extract tickers and their custom names from the configuration.
    
    Args:
        config: The loaded configuration dictionary
        
    Returns:
        Dict mapping categories to their ticker mappings
    """
    tickers = {}
    for category in ['sprds', 'derv']:
        tickers[category] = {
            security['custom_name']: security['ticker']
            for security in config[category]['securities']
        }
    return tickers

def get_fields_by_category(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract fields for each category from the configuration.
    
    Args:
        config: The loaded configuration dictionary
        
    Returns:
        Dict mapping categories to their fields
    """
    return {
        category: config[category]['field']
        for category in ['sprds', 'derv']
    }
