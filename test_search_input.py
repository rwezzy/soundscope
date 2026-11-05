"""The viewer can accept a question without firing robot/viewer shortcuts."""
import unittest
from scope import edit_search_input


class SearchInput(unittest.TestCase):
    def test_question_characters_keep_case_and_spaces(self):
        text = ''
        for char in 'When did my phone ring?':
            text, action = edit_search_input(text, ord(char))
            self.assertEqual(action, 'edit')
        self.assertEqual(text, 'When did my phone ring?')
        self.assertEqual(edit_search_input(text, 13), (text, 'submit'))

    def test_cancel_backspace_empty_submit_and_bound(self):
        self.assertEqual(edit_search_input('abc', 27), (None, 'cancel'))
        self.assertEqual(edit_search_input('abc', 8), ('ab', 'edit'))
        self.assertEqual(edit_search_input('  ', 13), ('', 'edit'))
        self.assertEqual(edit_search_input('a' * 240, 65), ('a' * 240, 'edit'))
        self.assertEqual(edit_search_input('abc', 255), ('abc', 'edit'))


if __name__ == '__main__':
    unittest.main()
