"""CPU-only parsing and Markdown tests for phase-boundary hardware samples."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
PATH=HERE/'bench.py' if (HERE/'bench.py').exists() else HERE.parent/'scripts/bench.py'
spec=importlib.util.spec_from_file_location('hardware_bench',PATH)
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

class HardwareTests(unittest.TestCase):
    def phase(self):
        def snapshot(offset):
            return {f'rank{i}':{'verified':True,'stdout':f'SM [MHz], Max SM [MHz], Temp [C], Power [W]\n{1800+100*i+offset} MHz, 3000 MHz, 45, {50+10*i+offset/10:.2f} W\ninactive\nGPU_RC=0 CAP_RC=3\n'} for i in range(4)}
        return {'name':'decode_s4','kind':'s4','clock_label':'fullclk',
                'clock_before':snapshot(0),'clock_after':snapshot(100),'runs':[]}
    def test_min_median_max(self):
        phase=self.phase();summary=bench.phase_hardware(phase)
        self.assertEqual(summary['sm_mhz']['n'],8)
        self.assertEqual(bench.hardware_cell(phase,'sm_mhz'),'1800 / 2000 / 2200')
        self.assertEqual(bench.hardware_cell(phase,'power_w'),'50.0 / 70.0 / 90.0')
    def test_missing_samples_not_claimed_as_full(self):
        phase=self.phase();phase['clock_after']['rank0']['verified']=False
        self.assertEqual(bench.hardware_cell(phase,'sm_mhz'),'N/A (7/8 samples)')
        phase=self.phase();phase['clock_after']['rank0']['stdout']='1800 MHz, 3000 MHz, 45, N/A\n'
        self.assertEqual(bench.hardware_cell(phase,'power_w'),'N/A (7/8 samples)')
    def test_s1_s4_tables_show_clock_and_power(self):
        phase=self.phase()
        s1={**phase,'name':'decode_prose','kind':'s1','runs':[{'case':{'warmup':False},'result':{'ttft_s':1,'decode_tok_s':30,'prompt_tok_s_proxy':100,'warnings':[]}}]}
        s4={**phase,'runs':[{'rep':1,'batch':{'aggregate_decode_tok_s':100,'per_stream_tok_s':[25]*4,'passed':True}}]}
        table=bench.markdown({'phases':[s1,s4]})
        self.assertEqual(table.count('1800 / 2000 / 2200'),2)
        self.assertEqual(table.count('50.0 / 70.0 / 90.0'),2)
        self.assertIn('boundary samples',table)

if __name__=='__main__':unittest.main(verbosity=2)
