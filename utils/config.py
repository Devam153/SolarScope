import os
class SolarConfig:
    """Indian-market defaults + the one external API key we need at boot."""

    # ---- API keys -----------------------------------------------------------
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

    # ---- Indian-market financial defaults -----------------------------------
    COST_PER_WATT_INSTALLED = 45        # ₹/W (2024 residential market avg)
    DEFAULT_ELECTRICITY_RATE = 6.50      # ₹/kWh (national residential avg)
    CENTRAL_SUBSIDY = 0.40               # 40% on first 3 kW (PM Surya Ghar approx)
    SYSTEM_LIFETIME = 25                 # years (financial horizon)

    @classmethod
    def validate_config(cls) -> dict:
        """Check the only API key the live pipeline actually needs."""
        issues = []
        if not cls.GOOGLE_MAPS_API_KEY:
            issues.append("GOOGLE_MAPS_API_KEY is not set")
        return {"valid": not issues, "issues": issues}

    @classmethod
    def format_currency(cls, amount: float) -> str:
        """Format ₹ in Indian convention: Cr / L / plain."""
        if amount >= 10_000_000:           # ≥ 1 crore
            return f"₹{amount / 10_000_000:.1f} Cr"
        if amount >= 100_000:              # ≥ 1 lakh
            return f"₹{amount / 100_000:.1f} L"
        return f"₹{amount:,.0f}"


# Global singleton — imported by app.py and components/pipeline.py
config = SolarConfig()


def validate_environment() -> dict:
    """App-startup gate: returns 'ready' if Google Maps key is configured."""
    validation = config.validate_config()
    if validation["valid"]:
        return {"status": "ready", "message": "Configuration OK"}
    return {
        "status": "error",
        "message": f"Configuration issues: {', '.join(validation['issues'])}",
    }
