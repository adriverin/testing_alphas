#!/usr/bin/env python3
"""
Simple HTTP Server for Trading Dashboard

This script starts a local HTTP server to serve the trading dashboard,
which avoids CORS issues when loading local JSON files.
"""

import http.server
import socketserver
import webbrowser
import sys
import os
from pathlib import Path

def start_server(port=8000, auto_open=True):
    """Start HTTP server and optionally open browser"""
    
    # Check if dashboard files exist
    html_file = Path('trading_strategy_dashboard.html')
    if not html_file.exists():
        print("❌ Error: trading_strategy_dashboard.html not found in current directory")
        print("   Make sure you're running this from the correct folder")
        return 1
    
    # Check if data file exists
    data_file = Path('dashboard_data.json')
    if not data_file.exists():
        print("⚠️  Warning: dashboard_data.json not found")
        print("   Dashboard will use sample data. To generate real data:")
        print("   python generate_dashboard_data.py")
        print()
    
    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers to allow local file access
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
        
        def log_message(self, format, *args):
            # Suppress log messages for cleaner output
            pass
    
    try:
        with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
            url = f"http://localhost:{port}/trading_strategy_dashboard.html"
            
            print(f"🚀 Starting HTTP server on port {port}")
            print(f"📊 Dashboard URL: {url}")
            print(f"🛑 Press Ctrl+C to stop the server")
            print("-" * 50)
            
            if auto_open:
                print("🌐 Opening dashboard in browser...")
                webbrowser.open(url)
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use")
            print(f"   Try a different port: python serve_dashboard.py --port {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")
        return 1

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Serve trading dashboard via HTTP")
    parser.add_argument('--port', type=int, default=8000, 
                       help='Port number (default: 8000)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Don\'t automatically open browser')
    
    args = parser.parse_args()
    
    return start_server(port=args.port, auto_open=not args.no_browser)

if __name__ == '__main__':
    sys.exit(main()) 