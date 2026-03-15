"""Shell completion script generators for cozempic."""

from __future__ import annotations


def bash_completion() -> str:
    """Generate bash completion script."""
    import cozempic.strategies  # noqa: F401 — ensure strategies registered
    from .registry import PRESCRIPTIONS, STRATEGIES

    subcommands = (
        "list current diagnose treat strategy reload checkpoint "
        "post-compact guard init doctor formulary completions"
    )
    prescriptions = " ".join(PRESCRIPTIONS.keys())
    strategies = " ".join(STRATEGIES.keys())

    return f'''# cozempic bash completion
_cozempic() {{
    local cur prev
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"

    if [[ $COMP_CWORD == 1 ]]; then
        COMPREPLY=($(compgen -W "{subcommands}" -- "$cur"))
        return
    fi

    case "$prev" in
        -rx) COMPREPLY=($(compgen -W "{prescriptions}" -- "$cur")) ;;
        strategy) COMPREPLY=($(compgen -W "{strategies}" -- "$cur")) ;;
        treat|diagnose) COMPREPLY=($(compgen -W "$(cozempic list 2>/dev/null | awk 'NR>2 {{print $1}}') current" -- "$cur")) ;;
        completions) COMPREPLY=($(compgen -W "bash zsh" -- "$cur")) ;;
        --thinking-mode) COMPREPLY=($(compgen -W "remove truncate signature-only" -- "$cur")) ;;
    esac
}}
complete -F _cozempic cozempic
'''


def zsh_completion() -> str:
    """Generate zsh completion script."""
    import cozempic.strategies  # noqa: F401
    from .registry import PRESCRIPTIONS, STRATEGIES

    subcommands = (
        "list current diagnose treat strategy reload checkpoint "
        "post-compact guard init doctor formulary completions"
    )
    prescriptions = " ".join(PRESCRIPTIONS.keys())
    strategies = " ".join(STRATEGIES.keys())

    return f'''#compdef cozempic
_cozempic() {{
    local -a subcommands
    subcommands=({subcommands})
    _arguments -C '1:command:compadd -a subcommands' '*::arg:->args'
    case $state in
        args)
            case $words[1] in
                treat|diagnose) _arguments '1:session:(current)' '-rx[Prescription]:rx:({prescriptions})' '--execute[Apply]' ;;
                strategy) _arguments '1:strategy:({strategies})' '2:session:(current)' '--execute[Apply]' ;;
                completions) _arguments '1:shell:(bash zsh)' ;;
            esac ;;
    esac
}}
_cozempic "$@"
'''
