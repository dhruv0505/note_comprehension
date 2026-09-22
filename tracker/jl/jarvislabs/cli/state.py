# Global CLI state for the active command process.
# Separate module to avoid circular imports between app.py and command modules.

json_output: bool = False
yes: bool = False
