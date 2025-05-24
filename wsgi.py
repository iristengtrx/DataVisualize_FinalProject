# wsgi.py

from app import app
import os

if __name__ == "__main__":
    # 注意：这里我们使用了环境变量 PORT 来动态设置端口
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)