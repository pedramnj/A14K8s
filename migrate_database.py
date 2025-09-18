#!/usr/bin/env python3
"""
Database Migration Script
Adds new authentication fields to the Server model
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append('.')

from ai_kubernetes_web_app import app, db, Server

def migrate_database():
    """Migrate the database to add new Server fields"""
    print("🔄 Starting database migration...")
    
    with app.app_context():
        try:
            # Check if new columns exist
            inspector = db.inspect(db.engine)
            existing_columns = [col['name'] for col in inspector.get_columns('server')]
            
            new_columns = [
                'username', 'password', 'ssh_key', 'ssh_key_path', 'ssh_port',
                'namespace', 'connection_timeout', 'verify_ssl',
                'last_connection_test', 'connection_error'
            ]
            
            missing_columns = [col for col in new_columns if col not in existing_columns]
            
            if not missing_columns:
                print("✅ Database is already up to date!")
                return
            
            print(f"📋 Adding {len(missing_columns)} new columns to Server table...")
            
            # Add columns using ALTER TABLE statements
            for column in missing_columns:
                if column == 'username':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN username VARCHAR(100)'))
                elif column == 'password':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN password VARCHAR(255)'))
                elif column == 'ssh_key':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN ssh_key TEXT'))
                elif column == 'ssh_key_path':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN ssh_key_path VARCHAR(255)'))
                elif column == 'ssh_port':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN ssh_port INTEGER DEFAULT 22'))
                elif column == 'namespace':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN namespace VARCHAR(100) DEFAULT "default"'))
                elif column == 'connection_timeout':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN connection_timeout INTEGER DEFAULT 30'))
                elif column == 'verify_ssl':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN verify_ssl BOOLEAN DEFAULT 1'))
                elif column == 'last_connection_test':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN last_connection_test DATETIME'))
                elif column == 'connection_error':
                    db.session.execute(db.text('ALTER TABLE server ADD COLUMN connection_error TEXT'))
            
            # Commit the schema changes
            db.session.commit()
            
            print("✅ Database migration completed successfully!")
            print(f"   Added columns: {', '.join(missing_columns)}")
            
            # Update existing servers with default values
            servers = Server.query.all()
            updated_count = 0
            
            for server in servers:
                needs_update = False
                
                if not hasattr(server, 'ssh_port') or server.ssh_port is None:
                    server.ssh_port = 22
                    needs_update = True
                
                if not hasattr(server, 'namespace') or server.namespace is None:
                    server.namespace = 'default'
                    needs_update = True
                
                if not hasattr(server, 'connection_timeout') or server.connection_timeout is None:
                    server.connection_timeout = 30
                    needs_update = True
                
                if not hasattr(server, 'verify_ssl') or server.verify_ssl is None:
                    server.verify_ssl = True
                    needs_update = True
                
                if needs_update:
                    db.session.add(server)
                    updated_count += 1
            
            if updated_count > 0:
                db.session.commit()
                print(f"✅ Updated {updated_count} existing servers with default values")
            
        except Exception as e:
            print(f"❌ Migration failed: {str(e)}")
            db.session.rollback()
            raise

if __name__ == "__main__":
    migrate_database()
