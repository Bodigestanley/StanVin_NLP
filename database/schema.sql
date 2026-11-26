CREATE TABLE IF NOT EXISTS datasets (
 id BIGSERIAL PRIMARY KEY, name VARCHAR(200) NOT NULL, source_name VARCHAR(200),
 licence TEXT, provenance_notes TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS dataset_records (
 id BIGSERIAL PRIMARY KEY, dataset_id BIGINT REFERENCES datasets(id) ON DELETE SET NULL,
 source_record_id VARCHAR(200), text_content TEXT NOT NULL, original_label VARCHAR(120),
 mapped_label VARCHAR(50) CHECK(mapped_label IN ('Sizing','Damaged Item','Wrong Product','Late Delivery')),
 annotation_status VARCHAR(30) DEFAULT 'pending', created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS model_versions (
 id BIGSERIAL PRIMARY KEY, model_name VARCHAR(120) NOT NULL, version VARCHAR(80) NOT NULL,
 artifact_path TEXT, notes TEXT, created_at TIMESTAMPTZ DEFAULT NOW(), UNIQUE(model_name,version)
);
CREATE TABLE IF NOT EXISTS predictions (
 id BIGSERIAL PRIMARY KEY, input_text TEXT NOT NULL, predicted_category VARCHAR(50) NOT NULL,
 confidence DOUBLE PRECISION, model_name VARCHAR(120), demo_fallback BOOLEAN DEFAULT FALSE,
 created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS evaluation_runs (
 id BIGSERIAL PRIMARY KEY, model_version_id BIGINT REFERENCES model_versions(id),
 dataset_id BIGINT REFERENCES datasets(id), accuracy DOUBLE PRECISION, macro_f1 DOUBLE PRECISION,
 weighted_f1 DOUBLE PRECISION, confusion_matrix JSONB, report_json JSONB, created_at TIMESTAMPTZ DEFAULT NOW()
);
