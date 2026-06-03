# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


source_spec = Path(SPECPATH) / "SAXS_Simulator_Optimized.spec"
source = source_spec.read_text(encoding="utf-8")
source = source.replace('name="SAXS_Simulator_Optimized"', 'name="SAXS_Simulator_Optimized_Debug"')
source = source.replace('console=False', 'console=True')
exec(compile(source, str(source_spec), "exec"))
