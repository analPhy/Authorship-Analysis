# Wikipedia Phrase Search App
# Wikipedia フレーズ検索アプリケーション

## Project Overview
## プロジェクト概要

This project is a simple web application using Python (Flask) for the backend and React (JavaScript) for the frontend. It allows users to fetch text from a specified Wikipedia URL, search for words or phrases (up to 2 words) within that text, and display the context of the matches.

It's designed as a learning project with several basic security measures implemented.

---

本プロジェクトは、Python (Flask) をバックエンド、React (JavaScript) をフロントエンドとして使用したシンプルなWebアプリケーションです。指定したWikipediaのページのテキストを取得し、その中から入力された単語やフレーズ（最大2単語）を検索し、周辺の文脈（コンテキスト）を表示します。

学習目的のプロジェクトであり、いくつかの基本的なセキュリティ対策を施しています。

## Features
## 機能

* Fetch text from a user-specified Wikipedia URL.
* Search for phrases of 1 or 2 words.
* Display search results with context snippets.
* Basic error handling for URL issues and search failures.
* Loading indicator during text processing and search.
* Includes basic security improvements (XSS prevention, URL validation, dev CORS, generic errors).
* **Note:** The application re-fetches and re-processes the page text on *every* search request, even for the same URL (no stateful cache is used for production compatibility).

---

* ユーザーがWikipediaのURLを入力できます。
* 入力されたURLのページのテキストを取得・解析します。
* テキスト内から最大2単語のフレーズを検索します。
* 検索に一致した箇所を文脈とともに表示します。
* URL取得時や検索時にエラーが表示されます。
* テキスト処理中はローディング表示が出ます。
* 基本的なセキュリティ対策済み（XSS対策、URL検証、CORS開発設定、汎用エラーなど）。
* **注意:** 同じURLでの複数回検索でも、毎回ページ取得・解析を行います（本番環境での状態管理対応のため、状態を持つキャッシュ機能は意図的に含めていません）。

## Requirements
## 動作環境

* Python 3.6+
* pip (Python package installer)
* Node.js (LTS recommended)
* npm or Yarn (Node.js package managers)

---

* Python 3.6以上
* pip (Pythonパッケージ管理ツール)
* Node.js (最新LTS推奨)
* npm または Yarn (Node.jsパッケージ管理ツール)

## Setup
## セットアップ

Follow these steps to set up the project on your local machine.

1.  **Place Project Files**
    * Place the `app.py` file in your project's root directory.
    * Place the React application files (like `package.json`, `src` folder, etc.) in a subdirectory named `frontend` within the root directory (e.g., `./frontend/package.json`, `./frontend/src/...`).
    * Adjust paths as necessary if your file structure is different.

2.  **Python Environment Setup (Backend)**
    * Open a terminal and navigate to your project's root directory.
    * Create a Python virtual environment.
        ```bash
        python -m venv .venv
        ```
    * Activate the virtual environment.
        * On macOS/Linux:
            ```bash
            source .venv/bin/activate
            ```
        * On Windows (Command Prompt):
            ```bash
            .venv\Scripts\activate.bat
            ```
        * On Windows (PowerShell):
            ```powershell
            .venv\Scripts\Activate.ps1
            ```
    * Install the required Python packages.
        ```bash
        pip install Flask Flask-Cors beautifulsoup4 nltk urllib3
        ```
    * Download NLTK's `punkt` data. The application might attempt to download it on the first run, but manual download ensures it's available.
        ```bash
        python -m nltk.downloader punkt
        ```
        * If prompted for a download directory, choose the default path (usually within your user's home directory).

3.  **Node.js Environment Setup (Frontend)**
    * In a terminal, navigate to the React application directory (`./frontend`).
        ```bash
        cd frontend
        ```
    * Install the required Node.js packages.
        ```bash
        npm install
        # Or if using Yarn
        # yarn install
        ```

---

以下の手順でプロジェクトをローカル環境にセットアップします。

1.  **プロジェクトファイルの配置**
    * `app.py` をプロジェクトのルートディレクトリに配置します。
    * Reactアプリケーションのファイル（`package.json`, `src` フォルダなど）を、ルートディレクトリ直下の `frontend` フォルダに配置します（例: `./frontend/package.json`, `./frontend/src/...`）。
    * お手元のファイル構成に合わせて適宜パスを読み替えてください。

2.  **Python環境のセットアップ（バックエンド）**
    * ターミナルを開き、プロジェクトのルートディレクトリに移動します。
    * Pythonの仮想環境を作成します。
        ```bash
        python -m venv .venv
        ```
    * 仮想環境をアクティベートします。
        * macOS/Linuxの場合:
            ```bash
            source .venv/bin/activate
            ```
        * Windowsの場合 (Command Prompt):
            ```bash
            .venv\Scripts\activate.bat
            ```
        * Windowsの場合 (PowerShell):
            ```powershell
            .venv\Scripts\Activate.ps1
            ```
    * 必要なPythonパッケージをインストールします。
        ```bash
        pip install Flask Flask-Cors beautifulsoup4 nltk urllib3
        ```
    * NLTKの `punkt` データをダウンロードします。アプリケーション実行時に自動で試みることもありますが、手動でダウンロードしておくと確実です。
        ```bash
        python -m nltk.downloader punkt
        ```
        * もしダウンロード先を聞かれたら、デフォルトのパス（通常はユーザーのホームディレクトリ内）を選択してください。

3.  **Node.js環境のセットアップ（フロントエンド）**
    * ターミナルで、Reactアプリケーションのディレクトリ (`./frontend`) に移動します。
        ```bash
        cd frontend
        ```
    * 必要なNode.jsパッケージをインストールします。
        ```bash
        npm install
        # もしくは Yarn を使用している場合
        # yarn install
        ```

## Running the Application
## アプリケーションの実行

Start the backend and frontend in separate terminals.

1.  **Start the Backend (Flask)**
    * In the terminal where your Python virtual environment is activated, ensure you are in the project's root directory.
    * Run the following command:
        ```bash
        python app.py
        ```
    * The server should start and display a message like:
        ```
        * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
        ```
    * Keep this terminal running the server.

2.  **Start the Frontend (React)**
    * Open a separate terminal and navigate to the React application directory (`./frontend`).
    * Run the following command:
        ```bash
        npm start
        # Or if using Yarn
        # yarn start
        ```
    * The React development server will start, and your browser should automatically open to `http://localhost:3000`.
    * Keep this terminal running the development server.

---

バックエンドとフロントエンドをそれぞれ別々のターミナルで起動します。

1.  **バックエンド（Flask）の起動**
    * Python仮想環境がアクティベートされているターミナルで、プロジェクトのルートディレクトリにいることを確認します。
    * 以下のコマンドを実行します。
        ```bash
        python app.py
        ```
    * サーバーが起動すると、以下のようなメッセージが表示されます。
        ```
        * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
        ```
    * このターミナルはサーバーを実行したままにしておきます。

2.  **フロントエンド（React）の起動**
    * 別のターミナルを開き、Reactアプリケーションのディレクトリ (`./frontend`) に移動します。
    * 以下のコマンドを実行します。
        ```bash
        npm start
        # もしくは Yarn を使用している場合
        # yarn start
        ```
    * React開発サーバーが起動し、ブラウザが自動で開きます。通常は `http://localhost:3000` でアクセスできます。
    * このターミナルも開発サーバーを実行したままにしておきます。

## Usage
## 使用方法

1.  Open your browser to `http://localhost:3000`.
2.  Enter the URL of a Wikipedia page you want to search in the "Wikipedia URL" input field (e.g., `https://en.wikipedia.org/wiki/Banana`).
3.  Enter the word or phrase (up to 2 words) you want to search for in the "Phrase" input field (e.g., `fruit` or `tropical fruit`).
4.  Click the "Search" button.
5.  The backend will fetch, parse, and search the text from the URL. The results will be displayed below the input fields.
6.  If an error occurs, a red error message will be shown.
7.  The button will show "Loading..." and be disabled while processing.

---

1.  React開発サーバーが起動したブラウザ (`http://localhost:3000`) を開きます。
2.  「Wikipedia URL」入力欄に検索したいWikipediaページのURLを入力します（例: `https://en.wikipedia.org/wiki/Banana`）。
3.  「Phrase」入力欄に検索したい単語またはフレーズ（最大2単語）を入力します（例: `fruit` または `tropical fruit`）。
4.  「Search」ボタンをクリックします。
5.  バックエンドがURLからテキストを取得・解析し、検索を実行します。結果はページ下部に表示されます。
6.  エラーが発生した場合は、赤色のメッセージが表示されます。
7.  テキスト取得・処理中はボタンが「Loading...」となり無効化されます。

## Notes on Production Deployment
## 本番環境へのデプロイに関する注意点

This application is configured for local development. For production deployment, you need to consider the following points and make appropriate settings/changes.

* **WSGI Server:** The Flask application must be run using a WSGI server like Gunicorn or uWSGI. The `if __name__ == "__main__":` block in `app.py` is for development only and is not executed in production.
* **Web Server:** It is common to use a web server like Nginx or Apache as a reverse proxy to serve static files (React build output) and forward API requests to the backend (Flask).
* **API Endpoint URL:** The `API_BASE_URL` constant in `frontend/src/App.tsx` (`http://localhost:5000/api`) needs to be changed to the actual URL of your deployed backend in production.
* **CORS Configuration:** The `allowed_origins` list in `app.py` must be strictly set to the domain(s) where your React application will be hosted. Keeping it as `"*"` in production is a security risk.
* **HTTPS:** Enable HTTPS for secure communication.
* **NLTK Data:** Ensure the NLTK `punkt` data is available in your server environment. It should typically be downloaded as part of your deployment process.
* **Logging:** Configure logging to output to files or a proper log aggregation system in production.
* **SSRF Mitigation:** While basic `.wikipedia.org` domain validation is included, consider if more robust SSRF protection (e.g., disallowing access to internal IP addresses, handling redirects) is necessary depending on your environment and threat model.

---

このREADMEは、プロジェクトの概要と起動手順を日英両方で提供します。必要に応じて、プロジェクトのバージョン管理、ライセンス情報、コントリビューションガイドなどのセクションを追加してください。