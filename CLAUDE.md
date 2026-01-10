# BABEL-BEATS - AI Context Document

## Project Overview
Language learning through AI-generated musical patterns

## Core Concept
- Speak the universal language of music
- AI-powered music generation tailored to specific use case
- Real-time data integration and processing
- User-centric design with clear value proposition

## Technical Architecture
- **Data Sources**: Language databases, Speech patterns, Cultural music libraries
- **AI Models**: 
  - Music generation: Specialized models for use case
  - Data processing: Custom preprocessing pipelines
  - Personalization: User preference learning
- **Frontend**: React/Next.js with appropriate UI/UX
- **Backend**: Python FastAPI, Redis, PostgreSQL
- **Infrastructure**: Cloud-native architecture

## MVP Features
1. Pronunciation rhythm mapping
2. Tonal language training
3. Cultural music integration
4. Basic web interface
5. Sharing and export functionality

## Grant Alignment
- **Innovation**: Novel application of AI in music
- **Impact**: Clear benefit to target audience
- **Scalability**: Cloud-native architecture
- **Sustainability**: Revenue model potential

## Development Status
- [ ] Step 1: Core concept validation
- [ ] Step 2: Technical architecture design
- [ ] Step 3: MVP feature set definition
- [ ] Step 4: Grant alignment mapping
- [ ] Step 5: Initial prototype code

## Key Decisions
- Focus on user value over technical complexity
- Start with core differentiating feature
- Build for scale from day one
- Prioritize user feedback loops

---

# Ralph Loop Development Methodology

## Autonomous Development Protocol

This project uses **Ralph Loop** - a continuous AI agent methodology for iterative development. Named after Ralph Wiggum, it embodies persistent iteration until completion.

### How It Works

1. Claude receives a task with clear completion criteria
2. Works autonomously, iterating until done
3. State persists in files and git, not context
4. Fresh context on each loop prevents pollution
5. Completion signaled via `<promise>COMPLETE</promise>`

### Starting a Ralph Loop

```bash
/ralph-loop "Your task description. Requirements: [list]. Output <promise>COMPLETE</promise> when done." --max-iterations 50
```

### Writing Effective Prompts

**Good prompts have:**
- Binary success criteria (testable pass/fail)
- Incremental phases for complex work
- Self-correction loops (run tests, fix, repeat)
- Escape hatches (`--max-iterations`)

**Example:**
```
Implement user authentication with JWT.

Phase 1: Create auth routes (register, login, logout)
Phase 2: Add JWT token generation and validation
Phase 3: Create auth middleware
Phase 4: Write tests (>80% coverage)
Phase 5: Update README with API docs

Run tests after each phase. Fix any failures before proceeding.
Output <promise>COMPLETE</promise> when all phases pass.
```

---

## Long-Form Autonomous Thinking

### When to Think Deeply

Use extended reasoning for:
- Architectural decisions affecting multiple files
- Debugging complex issues across the codebase
- Planning multi-step implementations
- Evaluating trade-offs between approaches

### Subagent Strategy

Launch specialized subagents for parallel work:

| Agent Type | Use Case |
|------------|----------|
| `Explore` | Codebase search, finding patterns |
| `Plan` | Architecture design, implementation strategy |
| `Bash` | Git operations, builds, deployments |
| `general-purpose` | Complex multi-step research |

**Parallel execution example:**
```
When researching a feature, launch:
- Explore agent: Find similar patterns in codebase
- Plan agent: Design implementation approach
- general-purpose agent: Research external docs
```

### Context Management

- State lives in files and git, not LLM memory
- Each loop iteration gets fresh context
- Use @fix_plan.md to track progress across sessions
- Commit frequently to preserve work

---

## Completion Criteria Standards

### For Features
```
Feature is COMPLETE when:
- [ ] All acceptance criteria met
- [ ] Tests pass (coverage >80%)
- [ ] No TypeScript errors
- [ ] Build succeeds
- [ ] README updated if needed
Output: <promise>COMPLETE</promise>
```

### For Bug Fixes
```
Bug fix is COMPLETE when:
- [ ] Root cause identified and documented
- [ ] Fix implemented
- [ ] Regression test added
- [ ] All existing tests pass
Output: <promise>COMPLETE</promise>
```

### For Refactoring
```
Refactor is COMPLETE when:
- [ ] Code restructured as specified
- [ ] No behavior changes (tests verify)
- [ ] All tests pass
- [ ] No new TypeScript errors
Output: <promise>COMPLETE</promise>
```

---

## Anti-Patterns to Avoid

1. **Vague goals** - "Make it better" has no exit condition
2. **Skipping tests** - No verification = no convergence
3. **Unlimited iterations** - Always set `--max-iterations`
4. **Monolithic tasks** - Break into phases
5. **Context pollution** - Let loops refresh context naturally

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `/ralph-loop "..."` | Start autonomous loop |
| `/cancel-ralph` | Stop current loop |
| `--max-iterations N` | Safety limit |
| `--completion-promise "X"` | Custom exit phrase |

---

*Ralph Loop methodology based on [Geoffrey Huntley's original technique](https://ghuntley.com/ralph/)*
