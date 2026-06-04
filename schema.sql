-- (r)elevē — run this in Supabase SQL Editor
-- https://supabase.com/dashboard/project/eqlcerxqppblzgjakutu/sql/new

create table if not exists sessions (
  id               text        primary key,
  created_at       timestamptz not null default now(),
  video_filename   text,
  video_url        text,
  exercise_id      text,
  exercise         text,
  score            integer,
  grade            text,
  corrections_count integer,
  duration_seconds numeric,
  report           jsonb
);

-- allow anon key to read and write (MVP — no auth yet)
alter table sessions enable row level security;

create policy "anon_all" on sessions
  for all
  using (true)
  with check (true);
