import os

import redis

# 连接到 Redis（密码从环境变量读取，避免把密码硬编码进代码）
client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
    password=os.getenv("REDIS_PASSWORD") or None,
)

# 测试读写
client.set("test_key", "Hello, Redis!")
value = client.get("test_key")
print(f"Redis value: {value}")
