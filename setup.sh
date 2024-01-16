
mkdir -p ~/.streamlit/


DEFAULT_EMAIL="your-email@domain.com"
DEFAULT_PORT=8501


read -p "Enter your email for Streamlit credentials [default: $DEFAULT_EMAIL]: " USER_EMAIL
USER_EMAIL=${USER_EMAIL:-$DEFAULT_EMAIL}

read -p "Enter the port for Streamlit [default: $DEFAULT_PORT]: " USER_PORT
USER_PORT=${USER_PORT:-$DEFAULT_PORT}


echo "[general]" > ~/.streamlit/credentials.toml
echo "email = \"$USER_EMAIL\"" >> ~/.streamlit/credentials.toml


echo "[server]" > ~/.streamlit/config.toml
echo "headless = true" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
echo "port = $USER_PORT" >> ~/.streamlit/config.toml

echo "Streamlit configuration files created successfully."
