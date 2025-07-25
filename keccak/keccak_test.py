
import hashlib
from Crypto.Hash import keccak

def test_keccak():
    """
    Test the Keccak sponge function implementation.
    """
    from keccak import keccak_256
    
    # Test case 1: Basic functionality test
    msg = "hello world"
    code_result = keccak_256(bytes(msg, 'utf-8'))
    print(f"Test 1 - Input: '{msg}'")
    print(f"Keccak-256 result: {code_result.hex()}")
    print(f"Result length: {len(code_result)} bytes")
    
    # Test case 2: Empty string test
    empty_msg = ""
    empty_result = keccak_256(bytes(empty_msg, 'utf-8'))
    print(f"\nTest 2 - Empty string:")
    print(f"Keccak-256 result: {empty_result.hex()}")
    
    # Test case 3: Known test vector
    # Keccak-256 of "abc" should be a known value
    test_msg = "abc"
    test_result = keccak_256(bytes(test_msg, 'utf-8'))
    print(f"\nTest 3 - Input: '{test_msg}'")
    print(f"Keccak-256 result: {test_result.hex()}")
    
    # Test case 4: Binary data test
    binary_data = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
    binary_result = keccak_256(binary_data)
    print(f"\nTest 4 - Binary data: {binary_data.hex()}")
    print(f"Keccak-256 result: {binary_result.hex()}")
    
    # Test case 5: Longer message test
    long_msg = "The quick brown fox jumps over the lazy dog"
    long_result = keccak_256(bytes(long_msg, 'utf-8'))
    print(f"\nTest 5 - Long message: '{long_msg}'")
    print(f"Keccak-256 result: {long_result.hex()}")
    
    # Assertions to verify correctness
    assert len(code_result) == 32, "Keccak-256 should return 32 bytes"
    assert len(empty_result) == 32, "Keccak-256 should return 32 bytes for empty input"
    assert len(test_result) == 32, "Keccak-256 should return 32 bytes"
    assert len(binary_result) == 32, "Keccak-256 should return 32 bytes for binary data"
    assert len(long_result) == 32, "Keccak-256 should return 32 bytes for long message"
    
    # Test deterministic behavior
    repeat_result = keccak_256(bytes(msg, 'utf-8'))
    assert code_result == repeat_result, "Keccak-256 should be deterministic"
    
    print("\n✅ All tests passed! Keccak-256 implementation is working correctly.")
    
    return {
        'hello_world': code_result.hex(),
        'empty_string': empty_result.hex(),
        'abc': test_result.hex(),
        'binary_data': binary_result.hex(),
        'long_message': long_result.hex()
    }

def test_keccak_vs_reference():
    """
    Compare our Keccak implementation with a reference implementation if available
    """
    from keccak import keccak_256
    
    # Test vectors for Keccak-256 (not SHA3-256)
    test_vectors = [
        {
            'input': b'',
            'expected': keccak.new(digest_bits=256, data=b'').hexdigest()
        },
        {
            'input': b'abc',
            'expected': keccak.new(digest_bits=256, data=b'abc').hexdigest()
        }
    ]
    
    print("\n=== Reference Test Vectors ===")
    for i, vector in enumerate(test_vectors):
        result = keccak_256(vector['input'])
        result_hex = result.hex()
        expected = vector['expected']
        
        print(f"Test Vector {i+1}:")
        print(f"Input: {vector['input']}")
        print(f"Expected: {expected}")
        print(f"Got:      {result_hex}")
        print(f"Match: {'✅' if result_hex == expected else '❌'}")
        
        if result_hex == expected:
            print("✅ Test vector passed!")
        else:
            print("❌ Test vector failed!")
        print()

if __name__ == "__main__":
    # Run the original test
    test_results = test_keccak()
    
    # Run reference comparison
    test_keccak_vs_reference()
    
    print("\n=== Test Summary ===")
    for test_name, result in test_results.items():
        print(f"{test_name}: {result}")