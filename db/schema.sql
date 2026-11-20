-- ShiftDesk schema — Postgres 16
-- All business tables include company_id; composite PKs include company_id.

CREATE TABLE IF NOT EXISTS companies (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    cutoff_minutes  INT  NOT NULL,          -- R1: must have this many minutes before scheduled_time
    max_shift_delta_minutes INT NOT NULL,   -- R2: |new_time - old_time| <= this
    service_start   TIME NOT NULL,          -- R3: earliest allowed new_time
    service_end     TIME NOT NULL,          -- R3: latest allowed new_time
    desk_phone      TEXT,
    transport_manager_id TEXT
);

CREATE TABLE IF NOT EXISTS employees (
    company_id  TEXT NOT NULL REFERENCES companies(id),
    employee_id TEXT NOT NULL,
    name        TEXT NOT NULL,
    PRIMARY KEY (company_id, employee_id)
);

CREATE TABLE IF NOT EXISTS routes (
    company_id  TEXT NOT NULL REFERENCES companies(id),
    route_id    TEXT NOT NULL,
    name        TEXT NOT NULL,
    PRIMARY KEY (company_id, route_id)
);

CREATE TABLE IF NOT EXISTS cabs (
    company_id      TEXT NOT NULL REFERENCES companies(id),
    cab_id          TEXT NOT NULL,
    route_id        TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('pickup', 'drop')),
    departure_time  TIME NOT NULL,          -- scheduled departure for this cab
    capacity        INT  NOT NULL,
    PRIMARY KEY (company_id, cab_id),
    FOREIGN KEY (company_id, route_id) REFERENCES routes(company_id, route_id)
);

CREATE TABLE IF NOT EXISTS roster_entries (
    company_id      TEXT NOT NULL REFERENCES companies(id),
    entry_id        TEXT NOT NULL,
    employee_id     TEXT NOT NULL,
    service_date    DATE NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('pickup', 'drop')),
    cab_id          TEXT NOT NULL,
    scheduled_time  TIME NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'cancelled', 'pending_change')),
    PRIMARY KEY (company_id, entry_id),
    FOREIGN KEY (company_id, employee_id) REFERENCES employees(company_id, employee_id),
    FOREIGN KEY (company_id, cab_id)      REFERENCES cabs(company_id, cab_id)
);

CREATE TABLE IF NOT EXISTS shift_change_requests (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id      TEXT NOT NULL REFERENCES companies(id),
    employee_id     TEXT NOT NULL,
    entry_id        TEXT NOT NULL,
    old_cab_id      TEXT NOT NULL,
    new_cab_id      TEXT,
    new_time        TIME,
    reason          TEXT,
    status          TEXT NOT NULL DEFAULT 'pending_approval'
                    CHECK (status IN (
                        'pending_approval', 'applied', 'rejected',
                        'expired', 'failed', 'approved_not_applied'
                    )),
    idempotency_key TEXT NOT NULL,
    decided_by      TEXT,
    decision_note   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at      TIMESTAMPTZ,
    UNIQUE (company_id, idempotency_key),
    FOREIGN KEY (company_id, employee_id) REFERENCES employees(company_id, employee_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    company_id  TEXT NOT NULL,
    actor_id    TEXT NOT NULL,
    actor_role  TEXT NOT NULL,
    action      TEXT NOT NULL,
    request_id  UUID,
    detail      JSONB
);

-- Append-only: prevent UPDATE and DELETE on audit_log.
CREATE OR REPLACE FUNCTION audit_log_no_mutate()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only: % is not allowed', TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS audit_log_immutable ON audit_log;
CREATE TRIGGER audit_log_immutable
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_no_mutate();

CREATE TABLE IF NOT EXISTS notifications (
    id          BIGSERIAL PRIMARY KEY,
    company_id  TEXT NOT NULL REFERENCES companies(id),
    employee_id TEXT NOT NULL,
    request_id  UUID,
    message     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    read        BOOLEAN NOT NULL DEFAULT false,
    FOREIGN KEY (company_id, employee_id) REFERENCES employees(company_id, employee_id)
);
