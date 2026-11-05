"""Saves your Viam machine's address and API key into meta-soundscope/.env (git-ignored).

  python set_viam_keys.py

Where to find them: app.viam.com > your machine > CONNECT > Python code sample > switch on
"Include API key". Paste each value when asked; quotes or the whole `api_key='...'` line are fine.
"""
import re
from pathlib import Path

ENV = Path(__file__).with_name('.env')


def ask(prompt):
    raw = input(prompt).strip()
    quoted = re.findall(r"""['"]([^'"]+)['"]""", raw)  # api_key='abc' -> abc
    return (quoted[-1] if quoted else raw).strip()


def main():
    print(__doc__.split('\n\n')[2].strip())
    address = ask('1/3  Address (ends in .viam.cloud): ')
    key_id = ask('2/3  API key ID (api_key_id): ')
    key = ask('3/3  API key (api_key): ')
    if not address.endswith('.viam.cloud') or not key or not key_id:
        raise SystemExit('That does not look right (the address ends in .viam.cloud). Nothing saved; try again.')
    keep = [line for line in (ENV.read_text(encoding='utf-8').splitlines() if ENV.exists() else [])
            if not line.startswith(('VIAM_ADDRESS=', 'VIAM_API_KEY=', 'VIAM_API_KEY_ID='))]
    keep += [f'VIAM_ADDRESS={address}', f'VIAM_API_KEY={key}', f'VIAM_API_KEY_ID={key_id}']
    ENV.write_text('\n'.join(keep) + '\n', encoding='utf-8')
    print(f'Saved to {ENV}. Your local Viam connection settings are ready.')


if __name__ == '__main__':
    main()
