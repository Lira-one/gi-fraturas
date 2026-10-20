"""Armazenamento persistente dos usuários do app (arquivo JSON local).

Projeto acadêmico, sem banco de dados real: os usuários ficam num JSON ao
lado do código (streamlit_app/data/users.json), para que o Gerente/Chefe
Clínico consiga adicionar e remover usuários pela própria interface e essas
mudanças persistam entre reinicializações do Streamlit. Senhas nunca são
guardadas em texto puro — apenas o hash SHA-256 (sem salt; em produção real
o indicado seria bcrypt/argon2 com salt por usuário).
"""
import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
USERS_FILE = DATA_DIR / "users.json"

# Único papel que pode ser criado/removido pela tela de administração do
# Gerente — ou seja, só contas "abaixo do nível Gerente".
MANAGED_ROLE = "radiologista"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _default_users() -> dict:
    return {
        "dra.ana": {
            "password_hash": hash_password("raiox2026"),
            "role": "radiologista",
            "display_name": "Dra. Ana Souza",
        },
        "gerente.carlos": {
            "password_hash": hash_password("gestao2026"),
            "role": "gerente",
            "display_name": "Carlos Pereira (Chefe Clínico)",
        },
    }


def load_users() -> dict:
    if not USERS_FILE.exists():
        users = _default_users()
        save_users(users)
        return users
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def add_managed_user(username: str, password: str, display_name: str) -> None:
    """Adiciona um novo usuário com papel 'radiologista' (único papel que o
    Gerente pode criar pelo app)."""
    username = username.strip()
    display_name = display_name.strip()
    if not username:
        raise ValueError("Informe um nome de usuário.")
    if not password:
        raise ValueError("Informe uma senha.")
    if not display_name:
        raise ValueError("Informe um nome de exibição.")

    users = load_users()
    if username in users:
        raise ValueError(f"O usuário '{username}' já existe.")

    users[username] = {
        "password_hash": hash_password(password),
        "role": MANAGED_ROLE,
        "display_name": display_name,
    }
    save_users(users)


def remove_managed_user(username: str) -> None:
    """Remove um usuário, desde que seu papel seja 'radiologista'. Contas de
    Gerente não podem ser removidas por aqui."""
    users = load_users()
    user = users.get(username)
    if user is None:
        raise ValueError(f"O usuário '{username}' não existe.")
    if user["role"] != MANAGED_ROLE:
        raise ValueError("Só é possível remover usuários com papel 'radiologista'.")
    del users[username]
    save_users(users)


def list_managed_users() -> dict:
    """Retorna apenas os usuários com papel 'radiologista' (os que o Gerente
    pode gerenciar pela interface)."""
    users = load_users()
    return {u: info for u, info in users.items() if info["role"] == MANAGED_ROLE}
