from dataclasses import dataclass
from socketserver import ThreadingMixIn
from types import NoneType
import xmlrpc.client
import pickle
import traceback
import inspect
from typing import List, TypeVar, Optional, Tuple, Callable
from functools import wraps
from xmlrpc.server import SimpleXMLRPCServer

from driftguard_ray.config import get_logger

logger = get_logger("rpc")

@dataclass
class Node:
    host: str = "http://127.0.0.1" # "0.0.0.0" or "http://127.0.0.1"
    port: int = 12000  # 11000
    
class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    """Multi-threaded XML-RPC server for federated learning coordination."""
    daemon_threads = True
    allow_reuse_address = True
    
T = TypeVar('T')

class RPCError(Exception):
    """RPC Error Class"""
    pass

def server_func(original_func: Callable) -> Callable:
    """
    Server-side RPC method decorator.
    
    - Original function should:
        - Accept a single Tuple argument.
        - Return a tuple.
    """
    sig = inspect.signature(original_func)
    parameters = list(sig.parameters.values())
    filtered = [p for p in parameters if p.name not in ('self', 'cls')]
    assert len(filtered) <= 1, (
        f"Tuple | None argument expected, got {len(filtered)} parameters"
    )

    @wraps(original_func)
    def wrapper(*args) -> xmlrpc.client.Binary | NoneType:
        """Wrapper function to handle RPC calls."""
    
        try:
            if filtered:
                args = list(args)
                args[-1] = pickle.loads(args[-1].data)  # -> binary to Tuple
                assert isinstance(args[-1], tuple | list), "Expected tuple argument"

            # result: Optional[xmlrpc.client.Binary] = None
            result = original_func(*args)
            assert isinstance(result, tuple | list | NoneType), "Expected tuple return value"
            return (
                xmlrpc.client.Binary(pickle.dumps(result)) if result else None
            )  # -> Tuple to binary
        except Exception:
            logger.error("RPC server exception:\n%s", traceback.format_exc())
            raise
        
    return wrapper

class RPCClient:
    """RPC Client for XML-RPC communication.
    Attributes:
        proxy: XML-RPC server proxy.
    """    
    def __init__(self, node: Node) -> None:
        """Initialize the RPC client with the server proxy."""
        self.proxy = xmlrpc.client.ServerProxy(f"{node.host}:{(node.port)}", allow_none=True)

    def call(self, original_func: Callable) -> Callable:
        """ Decorator to call a remote method via XML-RPC."""

        sig = inspect.signature(original_func)
        parameters = list(sig.parameters.values())
        filtered = [p for p in parameters if p.name not in ('self', 'cls')]
        assert len(filtered) <= 1, (
            f"Tuple | None argument expected, got {len(filtered)} parameters"
        )

        @wraps(original_func)
        def wrapper(*args) -> Tuple | List | NoneType:

            args = list(args)[len(args)-len(filtered):]  # exclude self/cls
            # 0. get remote method, required name matches the function name
            remote_method: Callable[[xmlrpc.client.Binary], xmlrpc.client.Binary] \
                = getattr(self.proxy, original_func.__name__)
                
            result_binary: Optional[xmlrpc.client.Binary] = None

            if args:
                assert isinstance(args[-1], tuple | list), "Expected tuple argument"
                args[-1] = xmlrpc.client.Binary(pickle.dumps(args[-1]))  # -> Tuple to binary
                
            result_binary = remote_method(*args)  # No arguments, call directly

            # 3. unpack the Binary result to a tuple
            result = pickle.loads(result_binary.data) if result_binary else None
            assert isinstance(result, tuple | list | NoneType), "Expected Tuple or None"
            return result

        return wrapper
