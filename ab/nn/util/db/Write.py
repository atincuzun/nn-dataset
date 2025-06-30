import json
import uuid

from ab.nn.util.Util import *
from ab.nn.util.db.Init import init_db, sql_conn, close_conn


def init_population():
    if not db_file.exists():
        init_db()
        json_n_code_to_db()


def code_to_db(cursor, table_name, code=None, code_file=None, force_name = None):
    # If the model does not exist, insert it with a new UUID
    if code_file:
        nm = code_file.stem
    elif force_name is None:
        nm = uuid4(code)
    else:
        nm = force_name
    if not code:
        with open(code_file, 'r') as file:
            code = file.read()
    id_val = uuid4(code)
    # Check if the model exists in the database
    cursor.execute(f"SELECT code FROM {table_name} WHERE name = ?", (nm,))
    existing_entry = cursor.fetchone()
    if existing_entry:
        # If model exists, update the code if it has changed
        existing_code = existing_entry[0]
        if existing_code != code:
            print(f"Updating code for model: {nm}")
            cursor.execute("UPDATE nn SET code = ?, id = ? WHERE name = ?", (code, id_val, nm))
    else:
        cursor.execute(f"INSERT INTO {table_name} (name, code, id) VALUES (?, ?, ?)", (_serialize_uid(nm), code, _serialize_uid(id_val)))
    return nm


def populate_code_table(table_name, cursor, name=None):
    """
    Populate the code table with models from the appropriate directory.
    """
    code_dir = nn_path(table_name)
    code_files = [code_dir / f"{name}.py"] if name else [Path(f) for f in code_dir.iterdir() if f.is_file() and f.suffix == '.py' and f.name != '__init__.py']
    for code_file in code_files:
        code_to_db(cursor, table_name, code_file=code_file)
    # print(f"{table_name} added/updated in the `{table_name}` table: {[f.stem for f in code_files]}")


# def populate_prm_table(table_name, cursor, prm, uid):
#     """
#     Populate the parameter table with variable number of parameters of different types.
#     """
#     for nm, value in prm.items():
#         cursor.execute(f"INSERT INTO {table_name} (uid, name, value, type) VALUES (?, ?, ?, ?)",
#                        (uid, nm, str(value), type(value).__name__))

def _serialize_value(value):
    """
    Return a DB-compatible scalar (str / int / float / bytes / None).
    """
    if isinstance(value, uuid.UUID):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bytes)):
        return value
    return str(value)


def _serialize_uid(uid):
    """
    Make sure the UID column always receives text (SQLite TEXT).
    """
    return str(uid) if isinstance(uid, uuid.UUID) else uid


def populate_prm_table(table_name, cursor, prm, uid):
    """
    Insert every hyper-parameter in its native Python type.
    The target table layout is (uid TEXT, name TEXT, value).
    """
    columns = getattr(populate_prm_table, "_cache", {}).get(table_name)
    if columns is None:
        columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")]
        populate_prm_table._cache = getattr(populate_prm_table, "_cache", {})
        populate_prm_table._cache[table_name] = columns

    has_type = "type" in columns

    for nm, value in prm.items():
        if has_type:
            cursor.execute(
                f"INSERT INTO {table_name} (uid, name, value, type) VALUES (?, ?, ?, ?)",
                (_serialize_uid(uid), nm, _serialize_value(value), type(value).__name__),
            )
        else:
            cursor.execute(
                f"INSERT INTO {table_name} (uid, name, value) VALUES (?, ?, ?)",
                (_serialize_uid(uid), nm, _serialize_value(value)),
            )


def save_stat(config_ext: tuple[str, str, str, str, int], prm, cursor):
    # Insert each trial into the database with epoch
    transform = prm['transform']
    uid = prm.pop('uid')
    extra_main_column_values = [prm.pop(nm, None) for nm in extra_main_columns]
    for nm in param_tables:
        populate_prm_table(nm, cursor, prm, uid)
    cursor.execute(f"""
        INSERT INTO stat (id, transform, prm, {', '.join(main_columns_ext + extra_main_columns)})
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (_serialize_uid(uid), transform, _serialize_uid(uid), *config_ext, *extra_main_column_values))


def json_n_code_to_db():
    """
    Reload all statistics into the database for all subconfigs and epochs.
    """
    conn, cursor = sql_conn()
    stat_base_path = Path(stat_dir)
    sub_configs = [d.name for d in stat_base_path.iterdir() if d.is_dir()]

    for sub_config_str in sub_configs:
        model_stat_dir = stat_base_path / sub_config_str

        for epoch_file in Path(model_stat_dir).iterdir():
            model_stat_file = model_stat_dir / epoch_file
            epoch = int(epoch_file.stem)

            with open(model_stat_file, 'r') as f:
                trials = json.load(f)

            for trial in trials:
                _, _, metric, nn = sub_config = conf_to_names(sub_config_str)
                populate_code_table('nn', cursor, name=nn)
                populate_code_table('metric', cursor, name=metric)
                populate_code_table('transform', cursor, name=trial['transform'])
                save_stat(sub_config + (epoch,), trial, cursor)
    close_conn(conn)
    print("All statistics reloaded successfully.")


def save_results(config_ext: tuple[str, str, str, str, int], prm: dict):
    """
    Save Optuna study results for a given model to SQLite DB
    :param config_ext: Tuple of names (Task, Dataset, Metric, Model, Epoch).
    :param prm: Dictionary of all saved parameters.
    """
    conn, cursor = sql_conn()
    save_stat(config_ext, prm, cursor)
    close_conn(conn)


def save_nn(nn_code: str, task: str, dataset: str, metric: str, epoch: int, prm: dict, force_name = None):
    conn, cursor = sql_conn()
    nn = code_to_db(cursor, 'nn', code=nn_code, force_name=force_name)
    save_stat((task, dataset, metric, nn, epoch), prm, cursor)
    close_conn(conn)
    return nn


init_population()
