"""Autenticação do app. Os usuários (e seus hashes de senha) são persistidos
por users_store.py; o Gerente/Chefe Clínico pode adicionar/remover contas de
Radiologista pela barra lateral (ver user_admin.py).
"""
from users_store import hash_password, load_users

ROLE_LABELS = {
    "radiologista": "Médico(a) Radiologista",
    "gerente": "Gerente / Chefe Clínico",
}


def authenticate(username: str, password: str):
    users = load_users()
    user = users.get(username)
    if user and user["password_hash"] == hash_password(password):
        return {
            "username": username,
            "role": user["role"],
            "display_name": user["display_name"],
        }
    return None
