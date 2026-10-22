#!/usr/bin/env python3
"""Freeze -> build without targets -> continue without targets -> open once.

Run `prepare` only after development tests. Every later scientific stage runs
the snapshotted code, checks the seal, and refuses to overwrite a block or open
an incomplete experiment. `status` never reads held-out outcomes. Smoke runs
use the OLD development goals and cannot support a confirmation claim.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'hpc/article1_skeleton')]
import metamath_native_skeleton as io
from metamath_native.proofs import Library, freeze, proof_text
from metamath_native.online_skeleton import OnlineConfig
from native_cumulative_policy import CumulativeGrowth, fragment_records
from compression_overlay import compress_above
from native_quality_evaluation import FrozenContinuation, CoverageMonitor, target_proof_text
from native_holdout import (METHODS, digest, make_catalogue, structural_profile,
    select_rules, rank_rules, merge_rules, poi_holdout, paired_inference, holm)

DEFAULT_INITIAL = ROOT/'output/metamath_native_study_20261020_v2/working_core_deep/library.json.gz'
DEFAULT_OLD = ROOT/'output/metamath_lemma_transactions_quality_20261020_v1'
DEFAULT_MODEL = ROOT/'output/article1_skeleton_gate_pilot_20261016_v1/training/prop_200.model.json'
DEFAULT_POI = ROOT/'output/poi_holdout_inputs_20261021_v1/poi_tptp.tgz'
PROCESS_ENV = dict(PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')


def scientific_environment():
    return dict(os.environ, **PROCESS_ENV)


def state(lib, allowed, source_hash):
    return dict(source_sha256=source_hash, basis=[asdict(x) for x in lib.basis],
                nodes=lib.json_nodes(), search_ids=sorted(set(allowed)))


def manifest(directory):
    return [dict(path=str(p.relative_to(directory)), sha256=io.sha(p))
            for p in sorted(directory.rglob('*')) if p.is_file()]


def check_files(directory, files):
    for r in files:
        path = directory/r['path']
        if not path.is_file() or io.sha(path) != r['sha256']:
            raise ValueError('Frozen file changed or missing: '+str(path))


def verify_seal(run, expected=None, include_holdout=False):
    seal_path = run/'SEAL.json'
    if expected is not None and io.sha(seal_path) != expected:
        raise ValueError('Seal does not match the explicitly supplied commitment')
    seal = json.loads(seal_path.read_text())
    # Blind workers hash only code/protocol/construction inputs. They do not
    # even read target bytes. The evaluator/auditor verifies those at opening.
    files = seal['files'] if include_holdout else [r for r in seal['files']
                                                 if not r['path'].startswith('evaluation/')]
    check_files(run, files)
    if seal['protocol_sha256'] != io.sha(run/'PROTOCOL.json'):
        raise ValueError('Protocol commitment changed')
    return json.loads((run/'PROTOCOL.json').read_text())


def snapshot_code(out):
    paths = [Path(__file__), ROOT/'scripts/metamath_native_skeleton.py',
             ROOT/'scripts/audit_native_holdout.py',
             *sorted((ROOT/'src/metamath_native').glob('*.py')),
             *sorted((ROOT/'hpc/article1_skeleton').glob('*.py')),
             ROOT/'tests/test_native_holdout.py']
    for path in paths:
        dest = out/'code'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)


def prepare(a):
    """Only corpus identities/counts, not any held-out method scores, appear."""
    parent=getattr(a,'technical_repair_of',None)
    amendment=None
    if parent is not None:
        if a.smoke:raise ValueError('A smoke run cannot replace a scientific attempt')
        parent=parent.resolve()
        prior=verify_seal(parent,io.sha(parent/'SEAL.json'))
        if any((parent/name).exists() for name in
               ('EVALUATION_OPENED.json','EVALUATION_RECEIPT.json','PIPELINE_COMPLETE.json')):
            raise ValueError('A scored/opened test cannot enter the unopened technical-repair route')
        failure=json.loads((parent/'construction/2026102105/FAILURE.json').read_text())
        if not (parent/'PIPELINE_FAILURE.json').is_file() or failure.get('message')!='Incorrect axiom provenance':
            raise ValueError('This amendment covers only the diagnosed blind-stage provenance rejection')
        amendment=dict(kind='disclosed_pre_scoring_technical_repair',
            previous_run=str(parent),previous_seal_sha256=io.sha(parent/'SEAL.json'),
            previous_failure_sha256=io.sha(parent/'PIPELINE_FAILURE.json'),
            previous_corpus_sha256=prior['corpus_sha256'],
            previous_targets_opened=False,
            authorised_change='Recompute imported descendant axiom origins from the actual remapped destination parents; do not weaken any proof check.',
            new_source_states='2026102201–2026102216; do not reuse any earlier blind trajectory',
            unchanged='Hypothesis, corpus, old model, inherited source, 217/16 catalogue, quota 12, five methods, two continuers, 60/90 CPU clocks, and success rule.',
            interpretation='A disclosed replacement after technical failure before first target scoring, not an independent replication or a new held-out corpus.')
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    for name in ('inputs', 'evaluation', 'construction'):
        (out/name).mkdir()
    snapshot_code(out)
    protected = {}
    for path in (a.initial, a.model, a.development/'FINAL_AUDIT.json',
                 a.development/'inputs/TARGET_POOL.json', a.poi, a.verifier):
        protected[str(path.resolve())] = io.sha(path)
    shutil.copyfile(a.initial, out/'inputs/initial.json.gz')
    shutil.copyfile(a.model, out/'inputs/model.json')
    shutil.copyfile(a.verifier, out/'inputs/metamath')
    (out/'inputs/metamath').chmod(0o755)
    initial = io.read_gzip(a.initial); lib = io.restored(initial)
    start = time.process_time()
    catalogue_lib, catalogue = make_catalogue(lib.basis)
    io.write_gzip(out/'inputs/catalogue.json.gz', state(catalogue_lib, [], initial['source_sha256']))
    io.write_json(out/'inputs/CATALOGUE.json', catalogue)
    cat_mm = out/'inputs/catalogue.mm'; cat_mm.write_text(proof_text(catalogue_lib)[0])
    cat_verify = io.independent_verify(out/'inputs/metamath', cat_mm, out/'inputs/catalogue.verify.log')
    catalogue_cpu = time.process_time()-start+cat_verify['child_cpu_seconds']
    # A strong FIXED control uses the same intrinsic rank, pooled over previous
    # development states. The held-out targets and new source states play no role.
    dev_protocol = json.loads((a.development/'PROTOCOL.json').read_text())
    pooled = Counter(); dev_inputs = []
    for seed in dev_protocol['seeds']:
        p = a.development/'blocks'/str(seed)/'inputs/early.source.json.gz'
        old = io.restored(io.read_gzip(p)); profile = structural_profile(old, start=len(lib.nodes))
        pooled.update(profile); dev_inputs.append(dict(path=str(p.resolve()), sha256=io.sha(p)))
        protected[str(p.resolve())] = io.sha(p)
    ranked, _ = rank_rules(catalogue, pooled)
    fixed = [i for i in ranked if i not in set(catalogue['short_ids'])][:a.quota]
    if len(fixed) != a.quota:
        raise ValueError('Insufficient independent catalogue rules')
    io.write_json(out/'inputs/FIXED_RULES.json', dict(ids=fixed, development_inputs=dev_inputs,
        ranking='Pooled intrinsic source-DAG profile; no human targets', profile=dict(pooled)))
    exposed = json.loads((a.development/'inputs/TARGET_POOL.json').read_text())
    io.write_json(out/'evaluation/EXPOSED_TARGETS.json', exposed)
    if a.smoke:
        targets = [dict(t, corpus_id=digest(t['body'])) for t in exposed[:24]]
        corpus = dict(phase='DEVELOPMENT SMOKE ONLY: reused exposed goals', retained_unique=len(targets))
    else:
        # Match the checksum published with the author's release. No tar file
        # or TPTP source gets executed. Every achieved goal still needs a proof.
        import hashlib
        if hashlib.md5(a.poi.read_bytes()).hexdigest() != 'c1d550c62e7429ffc6b27b4c36db8449':
            raise ValueError('POI archive differs from the author-published release')
        shutil.copyfile(a.poi, out/'evaluation/poi_tptp.tgz')
        targets, corpus = poi_holdout(a.poi, exposed)
        if corpus['archive_members'] != 1374:
            raise ValueError('POI release member count changed; review before freezing')
    io.write_json(out/'evaluation/TARGET_POOL.json', targets)
    io.write_json(out/'evaluation/CORPUS.json', corpus)
    seeds = ([2026102201+i for i in range(16)] if amendment else
             [2026102101+i for i in range(16)]) if not a.smoke else [999321, 999322]
    protocol = dict(schema='sealed-native-structural-confirmation/v1',
        phase='development_smoke' if a.smoke else 'prospective_once_only_confirmation',
        seeds=seeds, methods=list(METHODS), policies=['uniform', 'mixed'],
        quota=a.quota, maximum_mp=3,
        source_attempts=48 if a.smoke else 1500,
        execution_concurrency=1 if a.smoke else 2,
        formula_size=160, proof_height=256,
        future_cpu=.15 if a.smoke else 60., total_cpu=.25 if a.smoke else 90.,
        event_cap=100_000 if a.smoke else 1_000_000,
        source_initial_nodes=len(lib.nodes), source_sha256=initial['source_sha256'],
        catalogue_cold_cpu=catalogue_cpu, catalogue_verification=cat_verify,
        fixed_ids=fixed, protected=protected,
        targets=len(targets), corpus_sha256=io.sha(out/'evaluation/TARGET_POOL.json'),
        corpus=corpus,
        external_corpus=dict(author='Christoph Wernhard', name='POI',
            paper='https://arxiv.org/abs/2602.15511',
            url='http://cs.christophwernhard.com/cdtools/exp-thgen/downloads/poi_tptp.tgz',
            archive_sha256=io.sha(a.poi)),
        adaptation='A frozen heuristic ranks a richer common grammar using only newly built source proof fragments, their correctly substituted boundary equalities, and already observed parent reuse. No human goals, labels, test events, or target-trained reward.',
        fixed_control='Lock a quota-matched set ranked on old development DAGs; also retain the whole fixed extended grammar and a random quota control. All arms receive the COMPLETE <=2-MP set and identical inherited compressed core.',
        quota_scope='Quota limits added >=3-MP catalogue outputs in two matched controls and adaptive, NOT total skeleton size or compression threshold. Positive-marginal upper coding remains unlimited.',
        primary='For each source, take the common held-out residual uncovered by EVERY initial arm; average two neutral policies; count newly covered residual targets by 60 growth CPU-s. Only future performance, not initial coverage or build output, determines success.',
        primary_total='Repeat the SAME endpoint at 90 total cold CPU-s, including newly measured common source build/read, catalogue cold synthesis/verification (charged in full, not amortised), method preparation/selection/merge/compression/certification and continuation setup. Inherited historical library/model training remains excluded and disclosed.',
        timing_scope='Measured operational parent+verifier-child CPU, not wall time or all historical research costs. Interpreter startup, filesystem wall waits, offline held-out observer, terminal bulk future export and audit remain separately reported; never claim whole-job or from-axioms ATP acceleration.',
        comparisons=['fixed', 'random', 'fixed_all'],
        inference='16 paired source RNG blocks, not independent axiom systems; exact one-sided sign-flip tests assume exchangeable paired signs. Intersection of future and total tests (max p), Holm across three controls, alpha .05. No optional stopping.',
        success=dict(adjusted_p_less_than=.05, mean_extra_targets_at_least=1.,
            for_both_endpoints=True, adaptive_selection_distinct_at_least=2,
            all_planned_cells_required=True, all_horizons_required=True,
            all_certificates_and_permission_audit_required=True,
            all_three_controls_must_pass=True),
        secondary=['Exact matches, first-cover AUC and censoring',
                   'Initial and final beyond-basis coverage, payload size, all CPU components',
                   'Same-source selectivity and dependence on the source profile'],
        stopping='Fixed complete plan; any missing/corrupt cell, failure or early cap makes confirmation inconclusive. Do not silently drop or rerun failed cells. No test-driven rule/parameter change under this seal.',
        target_firewall='All source building and continuation finish, with hashed receipts, before evaluator opens goals. Workers never load evaluation/. Offline observer can only score committed events. This is a software/provenance firewall, NOT an OS security boundary.',
        limitations=['POI and the old corpus share Metamath ancestry; decontamination removes alpha-exact matches after specified definitions, not all equivalent or subsuming statements.',
            'One finite heuristic/corpus/basis/budget comparison cannot establish universal optimality, RL advantage, mathematical theorem novelty, scale-free emergence, or full ZFC coverage.',
            'Existing learned gate/model and native core remain unchanged; costs before the inherited checkpoint remain historical, so no from-axioms end-to-end speedup claim.',
            'An uninformative or zero-residual benchmark cannot support adaptive superiority. Keep a negative or inconclusive result.'],
        representation='The neutral mixed continuer includes the existing ID-dependent frontier sampler. This compares delivered interfaces under that fixed engine, not a representation-invariant semantic quality functional.',
        scientific_process_environment=PROCESS_ENV,
        runtime=dict(python=sys.version, platform=platform.platform()))
    if amendment:
        if protocol['corpus_sha256']!=amendment['previous_corpus_sha256']:
            raise ValueError('Technical repair must retain the same unopened target pool')
        for key in ('initial','model','development'):
            path=getattr(a,key)
            if key=='development':path=path/'FINAL_AUDIT.json'
            if prior['protected'].get(str(path.resolve()))!=io.sha(path):
                raise ValueError('Inherited scientific input changed: '+key)
        for key in ('quota','maximum_mp','source_attempts','execution_concurrency',
                    'formula_size','proof_height','future_cpu','total_cpu','event_cap',
                    'methods','policies','fixed_ids','comparisons','success'):
            if protocol[key]!=prior[key]:raise ValueError('Unapproved scientific change: '+key)
        protocol['phase']='prospective_technical_repair_before_first_target_opening'
        protocol['technical_repair']=amendment
        io.write_json(out/'TECHNICAL_AMENDMENT.json',amendment)
    io.write_json(out/'PROTOCOL.json', protocol)
    (out/'FROZEN_PLAN_RU.txt').write_text(
        'ОДНОКРАТНАЯ ПРОВЕРКА АДАПТИВНОГО СТРУКТУРНОГО ОТБОРА\n\n'
        'Гипотеза: прошлые повторно используемые фрагменты доказательств позволяют выбрать\n'
        'правила, с которыми замороженный скелет лучше продолжает вывод, чем с фиксированными\n'
        'правилами. Это гипотеза о конкретной эвристике и конечном бюджете, не об оптимуме.\n\n'
        '1. Сохраняем накопленное ядро, прежний допуск, очередь, отсечение активности\n'
        'и верхнее сжатие с положительной предельной выгодой. Общие короткие правила\n'
        'получают ВСЕ методы. Из более богатой грамматики адаптивный метод выбирает\n'
        '12 схем по прошлым фрагментам и их повторному использованию, без человеческих целей.\n'
        'Число 12 ограничивает добавленные выходы грамматики для сравнения ресурсов;\n'
        'оно не задаёт порог или общую степень сжатия скелета.\n\n'
        '2. Контроли: ядро с короткими правилами; 12 правил, заранее выбранных по старым\n'
        'источникам; случайные 12; вся фиксированная расширенная грамматика.\n'
        'Все получают те же исходные состояния, продолжатели и бюджеты.\n\n'
        f'3. Корпус: {len(targets)} уникальных целей. Фаза: {protocol["phase"]}.\n'
        'Для подтверждения используем POI после исключения прежних точных совпадений\n'
        'с раскрытием указанных определений. Общий источник Metamath ограничивает\n'
        'обобщаемость; это не независимая математическая традиция и не весь ZFC.\n\n'
        '4. Подтверждающий план: 16 новых состояний, 5 методов, 2 продолжателя.\n'
        'Они образуют 160 траекторий, НО ТОЛЬКО 16 парных статистических блоков.\n'
        'Основной результат — новый охват общего начально не покрытого остатка\n'
        'после 60 CPU-с дальнейшего роста; отдельно тот же охват при 90 CPU-с\n'
        'измеренной холодной операционной цены. Начальный охват не даёт победу.\n'
        'Старая история обучения и исходной библиотеки не пересчитана.\n'
        'Отдельно показываем наблюдатель, заключительный экспорт и аудит;\n'
        'это не ускорение всей исследовательской работы или доказательства с аксиом.\n\n'
        '5. Успех требует среднего прироста хотя бы на одну цель по каждому\n'
        'из двух отсчётов против КАЖДОГО из трёх основных контролей и p<0.05\n'
        'после поправки Холма. Парный перебор смены знаков требует обменности\n'
        'знаков при нулевой гипотезе. Две политики и вложенные отсчёты не увеличивают n.\n'
        'Если адаптивный отбор везде одинаков, гипотеза адаптивности не проходит.\n\n'
        '6. Сначала полностью завершаем слепой вывод и фиксируем хеши. Лишь затем\n'
        'оценщик один раз открывает цели. Независимый аудит заново проверяет доступ\n'
        'к леммам, первые достижения, целевые подстановки, сертификаты и статистику.\n'
        'Не меняем параметры после просмотра результата. Пропуски и технические\n'
        'ошибки не исключаем молча; они блокируют подтверждение.\n'
        'Слабый, нулевой или отрицательный результат сохраняем как результат.\n')
    if amendment:
        plan=out/'FROZEN_PLAN_RU.txt'
        plan.write_text('ТЕХНИЧЕСКАЯ ПОПРАВКА ДО ПЕРВОГО РАСКРЫТИЯ ЦЕЛЕЙ\n\n'
            'Предыдущий запечатанный опыт завершился отказом проверяющего ядра при импорте лемм.\n'
            'Его четыре удачных слепых блока не оцениваются и не переносятся сюда.\n'
            'Исправлено только вычисление списка аксиом после переназначения родителей.\n'
            'Новые состояния2026102201–2026102216; корпус, гипотеза, методы, бюджеты\n'
            'и критерии успеха сохранены. Это раскрытая техническая замена,\n'
            'НЕ независимая репликация и НЕ новая контрольная выборка.\n\n'+plan.read_text())
    suffix=('_technical_repair_of_'+amendment['previous_seal_sha256'] if amendment else '')
    registry = ROOT/'experiments/article1/native_holdout_commitments'/('POI_'+protocol['corpus_sha256']+suffix+'.json')
    if not a.smoke and registry.exists():
        raise ValueError('This holdout already has a confirmation commitment: '+str(registry))
    io.write_json(out/'SEAL.json', dict(schema='native-confirmation-seal/v1',
        protocol_sha256=io.sha(out/'PROTOCOL.json'), created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        files=manifest(out), note='Local immutable-content commitment, not an externally registered timestamp'))
    seal = io.sha(out/'SEAL.json')
    if not a.smoke:
        registry.parent.mkdir(parents=True, exist_ok=True)
        io.write_json(registry, dict(run=str(out), seal_sha256=seal,
            corpus_sha256=protocol['corpus_sha256'],
            rule='One confirmation for this corpus. A new hypothesis needs a new prospective validation, not an unlabelled rerun.'))
    (out/'RUN_RU.txt').write_text(
        'ЗАРАНЕЕ ЗАМОРОЖЕННАЯ ПРОВЕРКА\n\n'
        f'Фаза: {protocol["phase"]}\nSHA-256 SEAL.json: {seal}\n'
        f'План: {len(seeds)} источников × {len(METHODS)} методов × 2 продолжателя.\n'
        'Рабочие процессы не читают цели. Оценщик откроет их один раз после ВСЕХ ячеек.\n'
        'Проверка может опровергнуть преимущество метода; победа не гарантирована.\n\n'
        f'Весь цикл (до независимого аудита): {sys.executable} {out}/code/scripts/run_native_holdout.py execute --run {out} --seal {seal}\n'
        f'Только слепой этап: {sys.executable} {out}/code/scripts/run_native_holdout.py run --run {out} --seal {seal}\n'
        f'Статус: {sys.executable} {out}/code/scripts/run_native_holdout.py status --run {out}\n'
        f'Оценка после завершения: {sys.executable} {out}/code/scripts/run_native_holdout.py evaluate --run {out} --seal {seal}\n'
        f'Независимый аудит: {sys.executable} {out}/code/scripts/audit_native_holdout.py --run {out} --seal {seal} --out НОВАЯ_ПАПКА\n')
    print(json.dumps(dict(stage='sealed_without_test_scores', run=str(out), seal_sha256=seal,
        targets=len(targets), corpus=corpus, source_blocks=len(seeds), planned_cells=len(seeds)*10)), flush=True)


def certify(lib, directory, verifier, filename='library.mm'):
    start = time.process_time()
    proof = directory/filename; proof.write_text(proof_text(lib)[0])
    check = io.independent_verify(verifier, proof, directory/(filename+'.verify.log'))
    return time.process_time()-start, check


def build_source(run, protocol, seed, block):
    begin = time.process_time()
    initial = io.read_gzip(run/'inputs/initial.json.gz'); lib = io.restored(initial)
    model = json.loads((run/'inputs/model.json').read_text())
    cfg = OnlineConfig(attempts=protocol['source_attempts'],
        max_nodes=len(lib.nodes)+protocol['source_attempts']+1, proof_height=256)
    engine = CumulativeGrowth(lib, cfg, seed, model=model)
    result, _ = engine.run(policy='uniform')
    dictionary, codec = compress_above(fragment_records(lib),
        [n.id for n in lib.nodes if n.parents], engine.fragment_library)
    allowed = set(engine.live)|set(engine.fragment_library)|set(dictionary)
    # Snapshot source before any catalogue rule imports or continuation.
    io.write_gzip(block/'source.json.gz', state(lib, allowed, initial['source_sha256']))
    io.write_json(block/'SOURCE_COMPRESSION.json', codec)
    parent_cpu, check = certify(lib, block, run/'inputs/metamath', 'source.mm')
    common_cpu = time.process_time()-begin+check['child_cpu_seconds']
    row = dict(source_common_cpu=common_cpu, result=result,
        certificate_export_parent_cpu=parent_cpu, verification=check,
        inherited_history_cpu='not remeasured; identical supplied native checkpoint and old gate model',
        checkpoint_sha256=io.sha(block/'source.json.gz'))
    io.write_json(block/'SOURCE_RESULT.json', row)
    return row


def build_arm(run, protocol, seed, method, block, directory, source_row):
    start = time.process_time()
    old = io.read_gzip(block/'source.json.gz'); lib = io.restored(old)
    catalogue = json.loads((run/'inputs/CATALOGUE.json').read_text())
    catalogue_lib = io.restored(io.read_gzip(run/'inputs/catalogue.json.gz'))
    profile = (structural_profile(lib, start=protocol['source_initial_nodes'])
               if method == 'adaptive' else {})
    selected, selection = select_rules(catalogue, profile, protocol['fixed_ids'],
        method, protocol['quota'], seed+50_000_000)
    by_id = {r['id']:r for r in catalogue['rules']}
    rules = merge_rules(lib, catalogue_lib, [by_id[i]['node'] for i in selected])
    # Preserve the old payload. Code above it, do not silently rebuild from zero.
    dictionary, codec = compress_above(fragment_records(lib),
        [n.id for n in lib.nodes if n.parents], set(old['search_ids'])|set(rules))
    allowed = set(old['search_ids'])|set(rules)|set(dictionary)
    projected, interface, mapping = lib.project(sorted(allowed))
    roots = {n.id for n in projected.nodes if not n.parents}
    frozen = state(projected, set(interface)|roots, old['source_sha256'])
    io.write_gzip(directory/'initial.json.gz', frozen)
    io.write_json(directory/'SELECTION.json', dict(selection, selected_ids=selected,
        source_profile=profile, source_sha256=io.sha(block/'source.json.gz')))
    io.write_json(directory/'COMPRESSION.json', codec)
    parent_export, check = certify(projected, directory, run/'inputs/metamath')
    method_cpu = time.process_time()-start+check['child_cpu_seconds']
    row = dict(method=method, method_cold_cpu=method_cpu,
        common_source_cpu=source_row['source_common_cpu'],
        catalogue_cold_cpu=protocol['catalogue_cold_cpu'],
        total_before_continuation_cpu=method_cpu+source_row['source_common_cpu']+protocol['catalogue_cold_cpu'],
        projected_certificate_nodes=len(projected.nodes),
        initial_derived_anchors=len(set(interface)-roots),
        checkpoint_sha256=io.sha(directory/'initial.json.gz'),
        export_parent_cpu=parent_export, verification=check)
    io.write_json(directory/'BUILD.json', row)
    return row


def continue_blind(run, protocol, seed, policy, initial_path, build, directory):
    """No targets, no labels, no observer and no test-dependent stopping."""
    directory.mkdir(exist_ok=False)
    begin = time.process_time()
    initial = io.read_gzip(initial_path); lib = io.restored(initial)
    engine = FrozenContinuation(lib, initial['search_ids'], seed,
        formula_size=protocol['formula_size'], proof_height=protocol['proof_height'],
        max_events=protocol['event_cap'])
    setup = time.process_time()-begin
    horizon = max(protocol['future_cpu'], protocol['total_cpu'])
    while engine.growth_cpu < horizon and len(engine.events) < protocol['event_cap']:
        start = time.process_time(); event = engine.step(policy)
        engine.growth_cpu += time.process_time()-start
        if event is not None:
            event['cpu'] = engine.growth_cpu
    io.write_gzip(directory/'final.json.gz', state(lib, engine.available_ids(), initial['source_sha256']))
    io.write_gzip(directory/'events.json.gz', engine.events)
    io.write_gzip(directory/'witnesses.json.gz', [asdict(n) for n in engine.witnesses])
    export_parent, check = certify(engine.export_library(), directory, run/'inputs/metamath')
    row = dict(status='verified_target_blind', seed=seed, policy=policy,
        growth_cpu=engine.growth_cpu, setup_cpu=setup,
        total_before_growth_cpu=build['total_before_continuation_cpu']+setup,
        stop='cpu_budget' if engine.growth_cpu >= horizon else 'event_cap',
        completed_horizon=engine.growth_cpu >= horizon,
        admissions=len(engine.events), counters=dict(engine.counts),
        export_parent_cpu=export_parent, verification=check)
    io.write_json(directory/'RESULT.json', row)
    return row


def run_blocks(a):
    run = a.run.resolve(); protocol = verify_seal(run, a.seal)
    if (run/'EVALUATION_OPENED.json').exists():
        raise ValueError('Holdout already opened; new construction forbidden')
    seeds = protocol['seeds'] if a.seed is None else [a.seed]
    if any(s not in protocol['seeds'] for s in seeds):
        raise ValueError('Seed not in frozen plan')
    for seed in seeds:
        block = run/'construction'/str(seed)
        if (block/'RECEIPT.json').exists():
            receipt = json.loads((block/'RECEIPT.json').read_text())
            check_files(block, receipt['files'])
            print(json.dumps(dict(seed=seed, status='already_complete_and_hash_checked')), flush=True)
            continue
        block.mkdir(exist_ok=False)  # A partial/failed block is NOT silently rerun.
        try:
            source_row = build_source(run, protocol, seed, block)
            order = list(METHODS)
            shift = protocol['seeds'].index(seed) % len(order)
            order = order[shift:]+order[:shift]
            for method in order:
                directory = block/method; directory.mkdir()
                build = build_arm(run, protocol, seed, method, block, directory, source_row)
                for j, policy in enumerate(protocol['policies']):
                    continue_blind(run, protocol, seed+60_000_000+j*2_000_000, policy,
                        directory/'initial.json.gz', build, directory/policy)
                print(json.dumps(dict(seed=seed, method=method,
                    status='blind_trajectories_verified', anchors=build['initial_derived_anchors'])), flush=True)
            verify_seal(run, a.seal)
            io.write_json(block/'RECEIPT.json', dict(status='complete_before_target_opening',
                seed=seed, seal_sha256=a.seal, files=manifest(block)))
        except Exception as error:
            io.write_json(block/'FAILURE.json', dict(status='terminal_for_this_cell',
                error=type(error).__name__, message=str(error)))
            raise


def complete_receipts(run, protocol):
    receipts = []
    for seed in protocol['seeds']:
        block = run/'construction'/str(seed)
        path = block/'RECEIPT.json'
        if not path.exists():
            raise ValueError('Cannot open holdout before all frozen blocks finish: '+str(seed))
        row = json.loads(path.read_text()); check_files(block, row['files'])
        if row['status'] != 'complete_before_target_opening':
            raise ValueError('Invalid blind-stage receipt')
        receipts.append(dict(seed=seed, sha256=io.sha(path)))
    return receipts


def observed_times(targets, initial, final, events):
    """Observer runs OFFLINE only, after the blind-stage receipts commit."""
    monitor = CoverageMonitor(targets)
    for i in initial['search_ids']:
        monitor.observe(final, i, 0.)
    for event in events:
        monitor.observe(final, event['node'], event['cpu'])
    return monitor


def endpoint(monitor, residual, horizon, offset=0.):
    hit = sorted(i for i in residual if monitor.first_cover.get(i, float('inf'))+offset <= horizon)
    exact = sorted(i for i in residual if monitor.first_exact.get(i, float('inf'))+offset <= horizon)
    auc = sum(max(0., horizon-monitor.first_cover.get(i, float('inf'))-offset)
              for i in residual)/max(1., horizon*len(residual))
    return dict(new_residual=len(hit), residual_fraction=len(hit)/max(1, len(residual)),
                exact_new_residual=len(exact), residual_ids=hit, normalised_auc=auc)


def evaluate(a):
    run = a.run.resolve(); protocol = verify_seal(run, a.seal, include_holdout=True)
    receipts = complete_receipts(run, protocol)
    # Exclusive opening record precedes the first goal read. No second attempt
    # after observing results; independent audit may recompute, not retune.
    io.write_json(run/'EVALUATION_OPENED.json', dict(seal_sha256=a.seal,
        blind_receipts=receipts, opened_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    out = run/'evaluation/results'; out.mkdir(exist_ok=False)
    targets = [dict(t, body=freeze(t['body']), disjoint=freeze(t['disjoint']))
               for t in json.loads((run/'evaluation/TARGET_POOL.json').read_text())]
    rows = []; informative = True; complete = True; selections = set()
    for seed in protocol['seeds']:
        block = run/'construction'/str(seed); monitors = {}; initial_union = set()
        for method in METHODS:
            d = block/method; initial = io.read_gzip(d/'initial.json.gz')
            for policy in protocol['policies']:
                case = d/policy; final = io.restored(io.read_gzip(case/'final.json.gz'))
                events = io.read_gzip(case/'events.json.gz')
                monitor = observed_times(targets, initial, final, events)
                initial_union.update(i for i, t in monitor.first_cover.items() if t == 0.)
                monitors[method, policy] = monitor
        residual = set(range(len(targets)))-initial_union
        informative &= bool(residual)
        io.write_json(out/(str(seed)+'.RESIDUAL.json'), dict(initial_union=sorted(initial_union), residual=sorted(residual)))
        for method in METHODS:
            d = block/method
            if method == 'adaptive':
                selections.add(json.loads((d/'SELECTION.json').read_text())['selection_sha256'])
            for policy in protocol['policies']:
                case = d/policy; result = json.loads((case/'RESULT.json').read_text())
                monitor = monitors[method, policy]
                complete &= result['completed_horizon']
                dest = out/(str(seed)+'_'+method+'_'+policy); dest.mkdir()
                lib = io.restored(io.read_gzip(case/'final.json.gz'))
                text = target_proof_text(lib, monitor)
                proof = dest/'targets.mm'; proof.write_text(text)
                check = io.independent_verify(run/'inputs/metamath', proof, dest/'verify.log')
                io.write_json(dest/'HITS.json', dict(first_cover=monitor.first_cover,
                    first_exact=monitor.first_exact, covering=monitor.covering,
                    mappings={i:[(k,v) for k,v in m.items()] for i,m in monitor.mappings.items()},
                    observer_cpu=monitor.cpu, verification=check))
                future = endpoint(monitor, residual, protocol['future_cpu'])
                total = endpoint(monitor, residual, protocol['total_cpu'], result['total_before_growth_cpu'])
                rows.append(dict(seed=seed, method=method, policy=policy, residual_size=len(residual),
                    initial_coverage=sum(t == 0 for t in monitor.first_cover.values()),
                    future=future, total=total, observer_cpu=monitor.cpu))
    comparisons = {}; pvalues = {}
    for control in protocol['comparisons']:
        tests = {}
        for scope in ('future', 'total'):
            differences = []
            for seed in protocol['seeds']:
                def mean(method):
                    rs = [r for r in rows if r['seed'] == seed and r['method'] == method]
                    return sum(r[scope]['new_residual'] for r in rs)/len(rs)
                differences.append(mean('adaptive')-mean(control))
            tests[scope] = paired_inference(differences)
        pvalues[control] = max(tests['future']['p_one_sided'], tests['total']['p_one_sided'])
        comparisons[control] = tests
    adjusted = holm(pvalues)
    statistical = all(adjusted[c] < .05 and all(comparisons[c][s]['mean'] >= 1.
        for s in ('future', 'total')) for c in comparisons)
    eligible = informative and complete and len(selections) >= 2 and protocol['phase'] != 'development_smoke'
    conclusion = ('candidate_superiority_pending_independent_audit' if eligible and statistical
                  else 'no_confirmed_adaptive_superiority' if eligible else 'inconclusive_or_development_only')
    summary = dict(status=conclusion, independent_audit_required=True,
        eligible=eligible, all_horizons_complete=complete, informative_residual=informative,
        adaptive_distinct_selections=len(selections), source_blocks=len(protocol['seeds']),
        targets=len(targets), comparisons=comparisons, holm_adjusted_p=adjusted, rows=rows,
        scope='This frozen finite corpus/basis/budget only; no universal optimum, unseen mathematical lineage or priority claim',
        protocol_sha256=io.sha(run/'PROTOCOL.json'), seal_sha256=a.seal)
    io.write_json(out/'SUMMARY.json', summary)
    io.write_json(run/'EVALUATION_RECEIPT.json', dict(status='scored_once_pending_independent_audit',
        seal_sha256=a.seal, files=manifest(out)))
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}, ensure_ascii=False), flush=True)


def status(a):
    protocol = verify_seal(a.run)
    done = []; partial = []; failed = []
    for seed in protocol['seeds']:
        block = a.run/'construction'/str(seed)
        if (block/'FAILURE.json').exists(): failed.append(seed)
        elif (block/'RECEIPT.json').exists(): done.append(seed)
        elif block.exists(): partial.append(seed)
    print(json.dumps(dict(phase=protocol['phase'], blocks_complete=len(done),
        blocks_planned=len(protocol['seeds']), partial_or_running=partial,
        failed=failed, opened=(a.run/'EVALUATION_OPENED.json').exists(),
        evaluation_complete=(a.run/'EVALUATION_RECEIPT.json').exists(),
        seal_sha256=io.sha(a.run/'SEAL.json')), ensure_ascii=False, indent=2))


def execute(a):
    """Fixed parallel blind stage, then exactly one evaluator and one audit."""
    run = a.run.resolve(); protocol = verify_seal(run, a.seal)
    io.write_json(run/'PIPELINE_STARTED.json', dict(seal_sha256=a.seal,
        workers=protocol['execution_concurrency'],
        note='One fixed experiment, no interim held-out scores or outcome-based stopping'))
    logs = run/'logs'; logs.mkdir(exist_ok=False)
    script = run/'code/scripts/run_native_holdout.py'
    def worker(seed):
        with (logs/(str(seed)+'.log')).open('x') as log:
            subprocess.run([sys.executable, str(script), 'run', '--run', str(run),
                '--seal', a.seal, '--seed', str(seed)], stdout=log, stderr=subprocess.STDOUT, check=True)
        return seed
    pool = ThreadPoolExecutor(max_workers=protocol['execution_concurrency'])
    futures = []
    try:
        futures = [pool.submit(worker, seed) for seed in protocol['seeds']]
        for future in as_completed(futures):
            print(json.dumps(dict(stage='blind_block_complete', seed=future.result())), flush=True)
    except Exception as error:
        for future in futures: future.cancel()
        io.write_json(run/'PIPELINE_FAILURE.json', dict(error=type(error).__name__, message=str(error),
            action='Stop without scoring or automatic retry; inspect failed blind-stage logs.'))
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    subprocess.run([sys.executable, str(script), 'evaluate', '--run', str(run), '--seal', a.seal], check=True)
    subprocess.run([sys.executable, str(run/'code/scripts/audit_native_holdout.py'),
        '--run', str(run), '--seal', a.seal, '--out', str(run/'independent_review')], check=True)
    io.write_json(run/'PIPELINE_COMPLETE.json', dict(status='scored_and_independently_audited',
        seal_sha256=a.seal, audit_sha256=io.sha(run/'independent_review/INDEPENDENT_REVIEW.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--initial', type=Path, default=DEFAULT_INITIAL)
    prep.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    prep.add_argument('--development', type=Path, default=DEFAULT_OLD)
    prep.add_argument('--poi', type=Path, default=DEFAULT_POI)
    prep.add_argument('--verifier', type=Path, default=io.DEFAULT_VERIFIER)
    prep.add_argument('--quota', type=int, default=12); prep.add_argument('--smoke', action='store_true')
    prep.add_argument('--technical-repair-of',type=Path,
        help='Explicitly disclosed replacement of the diagnosed unopened technical failure; keeps the old commitment intact')
    for command in ('run', 'evaluate', 'execute', 'status'):
        parser = sub.add_parser(command); parser.add_argument('--run', type=Path, required=True)
        if command != 'status': parser.add_argument('--seal', required=True)
        if command == 'run': parser.add_argument('--seed', type=int)
    a = p.parse_args()
    if a.command != 'prepare' and a.command != 'status':
        # Execute the committed dependency tree, not mutable workspace imports.
        frozen = a.run.resolve()/'code/scripts/run_native_holdout.py'
        if (frozen.resolve() != Path(__file__).resolve()
                or any(os.environ.get(k) != v for k,v in PROCESS_ENV.items())):
            verify_seal(a.run.resolve(), a.seal)
            subprocess.run([sys.executable, str(frozen), *sys.argv[1:]], check=True,
                           env=scientific_environment())
            return
    {'prepare':prepare, 'run':run_blocks, 'execute':execute, 'evaluate':evaluate, 'status':status}[a.command](a)


if __name__ == '__main__':
    main()
