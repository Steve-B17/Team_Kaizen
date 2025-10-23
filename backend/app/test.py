# Test script
import asyncio
from rag_service import get_rag_answer

async def test():
    # Try with flights_info
    result1 = await get_rag_answer(
        "coimbatore to hyderabad flights", 
        "flights_info"
    )
    print("Result 1:", result1)
    
    # Try with flight_status
    result2 = await get_rag_answer(
        "coimbatore to hyderabad flights", 
        "flight_status"
    )
    print("Result 2:", result2)

asyncio.run(test())