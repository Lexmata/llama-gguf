# Makefile for llama-gguf
# Copyright (c) 2026 Lexmata LLC

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
MANDIR ?= $(PREFIX)/share/man

# Man pages
MAN1_PAGES := $(wildcard man/man1/*.1)

.PHONY: all build release install install-bin install-man uninstall clean help

all: build

build:
	cargo build

release:
	cargo build --release

# Install binary and man pages
install: install-bin install-man

# Install binary only
install-bin: release
	@echo "Installing llama-gguf to $(BINDIR)..."
	install -d $(BINDIR)
	install -m 755 target/release/llama-gguf $(BINDIR)/llama-gguf

# Install man pages only
install-man:
	@echo "Installing man pages to $(MANDIR)/man1..."
	install -d $(MANDIR)/man1
	@for page in $(MAN1_PAGES); do \
		echo "  Installing $$(basename $$page)"; \
		install -m 644 $$page $(MANDIR)/man1/; \
	done
	@echo "Man pages installed. Run 'mandb' if pages don't appear immediately."

# Uninstall binary and man pages
uninstall:
	@echo "Removing llama-gguf from $(BINDIR)..."
	rm -f $(BINDIR)/llama-gguf
	@echo "Removing man pages from $(MANDIR)/man1..."
	@for page in $(MAN1_PAGES); do \
		rm -f $(MANDIR)/man1/$$(basename $$page); \
	done
	@echo "Uninstall complete."

clean:
	cargo clean

# Show help
help:
	@echo "llama-gguf Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          Build debug version (default)"
	@echo "  build        Build debug version"
	@echo "  release      Build release version"
	@echo "  install      Install binary and man pages (requires sudo)"
	@echo "  install-bin  Install binary only"
	@echo "  install-man  Install man pages only"
	@echo "  uninstall    Remove installed files"
	@echo "  clean        Remove build artifacts"
	@echo "  help         Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  PREFIX       Installation prefix (default: /usr/local)"
	@echo "  BINDIR       Binary directory (default: \$$PREFIX/bin)"
	@echo "  MANDIR       Man page directory (default: \$$PREFIX/share/man)"
	@echo ""
	@echo "Examples:"
	@echo "  make release                  # Build release version"
	@echo "  sudo make install             # Install to /usr/local"
	@echo "  make PREFIX=~/.local install  # Install to ~/.local"
