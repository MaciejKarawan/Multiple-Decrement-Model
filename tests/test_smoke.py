from pathlib import Path
import ast, importlib, unittest
ROOT = Path(__file__).resolve().parents[1]
class SmokeTests(unittest.TestCase):
    def test_python_compiles(self):
        for directory in ['src','scripts']:
            for f in (ROOT/directory).rglob('*.py'):compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')

    def test_model_import_and_workbook_path(self):
        import decrement_model.model
        from decrement_model.paths import DATA
        self.assertTrue((DATA/'ttz2023.xlsx').is_file())
        import scripts.run_analysis

if __name__ == '__main__':unittest.main()
