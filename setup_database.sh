#!/bin/bash
# setup_database.sh - PostgreSQL and PGVector setup script

set -e

echo "🚀 Setting up PostgreSQL with PGVector for RAG system..."

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL is not installed. Please install it first:"
    echo "   sudo apt update && sudo apt install postgresql postgresql-contrib"
    exit 1
fi

# Default values
DB_NAME="rag_db"
DB_USER="rag_user"
DB_PASSWORD="rag_password"
DB_HOST="localhost"
DB_PORT="5432"

# Allow user to override defaults
read -p "Database name (default: rag_db): " input_db_name
DB_NAME=${input_db_name:-$DB_NAME}

read -p "Database user (default: rag_user): " input_db_user
DB_USER=${input_db_user:-$DB_USER}

read -p "Database password (default: rag_password): " input_db_password
DB_PASSWORD=${input_db_password:-$DB_PASSWORD}

echo "📊 Creating database and user..."

# Create database and user
sudo -u postgres psql << EOF
-- Drop database if exists (be careful in production!)
DROP DATABASE IF EXISTS $DB_NAME;
DROP USER IF EXISTS $DB_USER;

-- Create database and user
CREATE DATABASE $DB_NAME;
CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;

-- Connect to new database and enable vector extension
\c $DB_NAME
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant permissions on the database
GRANT ALL ON SCHEMA public TO $DB_USER;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $DB_USER;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $DB_USER;

-- Test the extension
SELECT * FROM pg_extension WHERE extname = 'vector';
EOF

echo "✅ Database setup completed!"

# Create .env file
ENV_FILE=".env"
echo "📝 Creating $ENV_FILE file..."

cat > $ENV_FILE << EOF
# PostgreSQL connection for RAG system
DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME

# Optional: Ollama configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
EOF

echo "✅ Environment file created: $ENV_FILE"

# Test connection
echo "🔍 Testing database connection..."
export DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"

python3 << 'EOF'
import sys
try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    
    # Test connection
    import os
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    register_vector(conn)
    
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"✅ PostgreSQL connection successful: {version}")
        
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        extension = cur.fetchone()
        if extension:
            print(f"✅ PGVector extension installed: {extension[1]}")
        else:
            print("❌ PGVector extension not found")
            sys.exit(1)
    
    conn.close()
    print("✅ Database setup verification completed!")
    
except ImportError as e:
    print(f"❌ Missing Python packages: {e}")
    print("Please install: pip install psycopg2-binary pgvector")
    sys.exit(1)
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    sys.exit(1)
EOF

echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Install Python dependencies: pip install -r requirements.txt"
echo "2. Start Ollama: ollama serve"
echo "3. Pull a model: ollama pull llama2"
echo "4. Start the app: uvicorn main:app --reload"
echo ""
echo "Your DATABASE_URL: postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
