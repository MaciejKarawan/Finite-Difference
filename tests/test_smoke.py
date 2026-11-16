from pathlib import Path
import ast, importlib, unittest
ROOT = Path(__file__).resolve().parents[1]
class SmokeTests(unittest.TestCase):
    def test_python_compiles(self):
        for directory in ['src','scripts']:
            for f in (ROOT/directory).rglob('*.py'):compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')

    def test_solver_imports_and_payoff(self):
        from finite_difference.pricing import option_price
        import finite_difference.fd, finite_difference.fd_ko, finite_difference.fd_ko_dividend
        self.assertEqual(option_price(120,100,0,0.05,0.2,'call'),20)

if __name__ == '__main__':unittest.main()
