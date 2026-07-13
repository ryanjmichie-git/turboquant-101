# Filler corpus

`corpus.txt` is ~1.2 MB (~300K tokens) of neutral long-form English prose:
Charles Darwin's *The Voyage of the Beagle* (1839), which is in the public
domain. The text was downloaded from Project Gutenberg (ebook #944) and the
Project Gutenberg header/footer boilerplate was removed, per their policy
for redistributing the plain public-domain text.

It exists so the needle-in-a-haystack benchmark works offline with zero
setup. To re-fetch or swap in a different text, run:

    python scripts/fetch_corpus.py
