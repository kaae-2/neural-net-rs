
default:
    @just --list --list-heading $'Available commands:\n'

alias t := test
alias e := example

test test_name='': 
    if [ {{test_name}} = '' ]; then \
    cargo test ; \
    else \
    cargo watch -q -c -x "test {{test_name}}"; \
    fi

example ex_file_name:
    cargo watch -q -c -x "run -q --example {{ex_file_name}}"