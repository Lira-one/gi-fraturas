"""Bloco de administração de usuários exibido na barra lateral, somente para
o papel Gerente/Chefe Clínico: permite adicionar e remover contas de
Radiologista (um nível abaixo do Gerente). Contas de Gerente não podem ser
criadas nem removidas por esta tela.
"""
import streamlit as st

from users_store import add_managed_user, list_managed_users, remove_managed_user


def render_user_admin():
    st.divider()
    with st.expander("👤 Gerenciar usuários (Radiologistas)"):
        st.caption(
            "Adicione ou remova contas de Radiologista. Contas de Gerente "
            "não podem ser alteradas por aqui."
        )

        st.markdown("**Adicionar Radiologista**")
        with st.form("add_user_form", clear_on_submit=True):
            new_username = st.text_input("Usuário")
            new_display_name = st.text_input("Nome de exibição")
            new_password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Adicionar", width="stretch")
        if submitted:
            try:
                add_managed_user(new_username, new_password, new_display_name)
                st.success(f"Usuário '{new_username}' adicionado.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

        st.markdown("**Radiologistas cadastrados**")
        managed_users = list_managed_users()
        if not managed_users:
            st.caption("Nenhum usuário Radiologista cadastrado.")
        for username, info in managed_users.items():
            col_name, col_remove = st.columns([3, 1])
            col_name.write(f"{info['display_name']}  \n`{username}`")
            if col_remove.button("Remover", key=f"remove_user_{username}"):
                try:
                    remove_managed_user(username)
                    st.success(f"Usuário '{username}' removido.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
