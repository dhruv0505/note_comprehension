"""SDK client — public Python API for JarvisLabs GPU cloud.

Usage:
    from jarvislabs import Client

    client = Client(api_key="...")
    instances = client.instances.list()
"""

from __future__ import annotations

from jarvislabs.account import Account
from jarvislabs.config import resolve_token
from jarvislabs.deployments import Deployments
from jarvislabs.exceptions import AuthError
from jarvislabs.filesystems import Filesystems
from jarvislabs.instances import Instances
from jarvislabs.scripts import Scripts
from jarvislabs.ssh_keys import SSHKeys
from jarvislabs.transport import Transport
from jarvislabs.vpcs import Vpcs


class Client:
    """Entry point for the JarvisLabs SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        token = resolve_token(api_key)
        if not token:
            raise AuthError("No API key found. Set JL_API_KEY or run: jl setup")
        self._transport = Transport(token)
        self.account = Account(self._transport)
        self.ssh_keys = SSHKeys(self._transport)
        self.scripts = Scripts(self._transport)
        self.filesystems = Filesystems(self._transport)
        self.vpcs = Vpcs(self._transport)
        self.instances = Instances(self._transport, self.ssh_keys)
        self.deployments = Deployments(self._transport)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
