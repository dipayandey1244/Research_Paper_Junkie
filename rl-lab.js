const ARMS = [.25, .45, .55, .65];
const MOVES = [[-1, 0], [0, 1], [1, 0], [0, -1]];
const LABELS = ['↑', '→', '↓', '←'];
const START = 0;
const GOAL = 15;
const HAZARDS = new Set([5, 10]);
const SAFE_STARTS = Array.from({length: GOAL}, (_, state) => state).filter(state => !HAZARDS.has(state));
const DISCOUNT = .95;

export function randomSource(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return (state + .5) / 4294967296;
  };
}

function bestIndex(values, random) {
  const best = Math.max(...values);
  const ties = values.map((value, index) => value === best ? index : -1).filter(index => index >= 0);
  return random ? ties[Math.floor(random() * ties.length)] : ties[0];
}

function betaSample(successes, failures, random) {
  // Beta(a,b) is the a-th order statistic of a+b−1 independent uniforms for integer a,b.
  const samples = Array.from({ length: successes + failures + 1 }, random);
  samples.sort((a, b) => a - b);
  return samples[successes];
}

export function runBandit(rounds, seed, epsilon, coefficient) {
  return ['Greedy', 'ε-greedy', 'UCB', 'Thompson'].map((name, strategy) => {
    const choose = randomSource(seed + strategy * 1009);
    const rewards = ARMS.map((_, index) => randomSource(seed ^ (0x9e3779b9 * (index + 1))));
    const counts = ARMS.map(() => 0);
    const wins = ARMS.map(() => 0);
    const curve = [0];
    for (let time = 1; time <= rounds; time++) {
      const means = counts.map((count, index) => count ? wins[index] / count : 0);
      let arm;
      if (strategy === 2) {
        const scores = means.map((mean, index) => counts[index] ?
          mean + coefficient * Math.sqrt(Math.log(time) / counts[index]) : Infinity);
        arm = bestIndex(scores, choose);
      } else if (strategy === 3) {
        arm = bestIndex(wins.map((win, index) => betaSample(win, counts[index] - win, choose)), choose);
      } else if (strategy === 1 && choose() < epsilon) {
        arm = Math.floor(choose() * ARMS.length);
      } else {
        arm = bestIndex(means, choose);
      }
      counts[arm]++;
      if (rewards[arm]() < ARMS[arm]) wins[arm]++;
      curve.push(curve.at(-1) + (Math.max(...ARMS) - ARMS[arm]));
    }
    return { name, counts, curve, reward: wins.reduce((a, b) => a + b, 0) };
  });
}

export function transition(state, action) {
  const [dr, dc] = MOVES[action];
  const row = Math.max(0, Math.min(3, Math.floor(state / 4) + dr));
  const column = Math.max(0, Math.min(3, state % 4 + dc));
  const next = row * 4 + column;
  const done = next === GOAL || HAZARDS.has(next);
  return { state, action, next, done, reward: next === GOAL ? 1 : HAZARDS.has(next) ? -1 : -.04 };
}

export function tabular(seed) {
  const table = Array.from({length: 16}, () => Array(4).fill(0));
  const random = randomSource(seed);
  let state = START;
  let steps = 0;
  let episodes = 0;
  let last;
  function step() {
    const action = random() < .25 ? Math.floor(random() * 4) : bestIndex(table[state], random);
    const experience = transition(state, action);
    const {reward, done, next} = experience;
    const before = table[state][action];
    const target = reward + (done ? 0 : DISCOUNT * Math.max(...table[next]));
    const after = before + .35 * (target - before);
    table[state][action] = after;
    last = {...experience, before, after, target};
    state = next;
    if (done || ++steps === 35) {
      state = START;
      steps = 0;
      episodes++;
    }
  }
  return {
    values: state => table[state], step,
    train(count) {
      const end = episodes + count;
      while (episodes < end) step();
    },
    get state() { return state; },
    get episodes() { return episodes; },
    get last() { return last; }
  };
}

function network(random) {
  const widths = [16, 16, 12, 4];
  return widths.slice(1).map((size, layer) => Array.from({length: size}, () =>
    Array.from({length: widths[layer] + 1}, (_, index) => index === widths[layer] ? 0 :
      (random() * 2 - 1) * Math.sqrt(6 / (widths[layer] + size)))));
}

function forward(model, state) {
  const activations = [Array.from({length: 16}, (_, index) => Number(index === state))];
  model.forEach((layer, index) => {
    const input = activations.at(-1);
    activations.push(layer.map(weights => {
      const value = input.reduce((sum, x, i) => sum + x * weights[i], weights.at(-1));
      return index === model.length - 1 ? value : Math.max(0, value);
    }));
  });
  return activations;
}

function addGradient(model, gradients, activations, action, error) {
  let deltas = Array(4).fill(0);
  // Derivative of Huber loss with threshold 1, for the selected action.
  deltas[action] = Math.max(-1, Math.min(1, error));
  for (let layer = model.length - 1; layer >= 0; layer--) {
    const input = activations[layer];
    const previous = input.map(() => 0);
    model[layer].forEach((weights, unit) => {
      input.forEach((value, i) => {
        gradients[layer][unit][i] += deltas[unit] * value;
        previous[i] += deltas[unit] * weights[i];
      });
      gradients[layer][unit][input.length] += deltas[unit];
    });
    deltas = previous.map((value, i) => input[i] > 0 ? value : 0);
  }
}

function copyModel(model) {
  return model.map(layer => layer.map(unit => [...unit]));
}

export function deepQ(seed) {
  const random = randomSource(seed);
  const model = network(random);
  let target = copyModel(model);
  const replay = [];
  let cursor = 0;
  let updates = 0;
  let episodes = 0;
  let meanLoss = null;
  const batchSize = 16;
  const values = state => forward(model, state).at(-1);

  function optimize() {
    const gradients = model.map(layer => layer.map(unit => unit.map(() => 0)));
    let loss = 0;
    for (let sample = 0; sample < batchSize; sample++) {
      const {state, action, reward, done, next} = replay[Math.floor(random() * replay.length)];
      const expected = reward + (done ? 0 : DISCOUNT * Math.max(...forward(target, next).at(-1)));
      const activations = forward(model, state);
      const error = activations.at(-1)[action] - expected;
      const magnitude = Math.abs(error);
      loss += magnitude <= 1 ? error ** 2 / 2 : magnitude - .5;
      addGradient(model, gradients, activations, action, error);
    }
    model.forEach((layer, l) => layer.forEach((unit, j) =>
      unit.forEach((weight, k) => { unit[k] = weight - .065 * gradients[l][j][k] / batchSize; })));
    meanLoss = loss / batchSize;
    if (++updates % 40 === 0) target = copyModel(model);
  }

  return {
    values,
    train(count) {
      const end = episodes + count;
      for (; episodes < end; episodes++) {
        let state = random() < .5 ? START : SAFE_STARTS[Math.floor(random() * SAFE_STARTS.length)];
        const epsilon = Math.max(.15, .98 ** episodes);
        for (let step = 0; step < 35; step++) {
          const action = random() < epsilon ? Math.floor(random() * 4) : bestIndex(values(state), random);
          const experience = transition(state, action);
          replay[cursor] = experience;
          cursor = (cursor + 1) % 512;
          if (replay.length >= 24) optimize();
          if (experience.done) break;
          state = experience.next;
        }
      }
    },
    get episodes() { return episodes; },
    get replaySize() { return replay.length; },
    get updates() { return updates; },
    get meanLoss() { return meanLoss; }
  };
}

export function rollout(values) {
  const route = [START];
  for (let step = 0; step < 35; step++) {
    const state = route.at(-1);
    const {next, done} = transition(state, bestIndex(values(state)));
    const repeated = route.includes(next);
    route.push(next);
    if (done || repeated) {
      return { route, result: next === GOAL ? 'Goal reached' : done ? 'Trap reached' : 'Policy loops' };
    }
  }
  return {route, result: 'Step limit'};
}

const $ = id => document.getElementById(id);
const readNumber = id => Number($(id).value);
const colors = ['#38435d', '#2459c4', '#09725c', '#a34520'];
const lineStyles = ['solid', 'dashed', 'dotted', 'dash-dot'];

function drawBandits(lines, rounds) {
  const canvas = $('bandit-chart');
  const ink = canvas.getContext('2d');
  const {width, height} = canvas;
  const left = 64, bottom = height - 47, top = 24, right = width - 18;
  const maximum = Math.max(1, ...lines.map(line => line.curve.at(-1)));
  ink.clearRect(0, 0, width, height);
  ink.font = '19px system-ui';
  ink.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = bottom - (bottom - top) * i / 4;
    ink.strokeStyle = '#dee5f0';
    ink.beginPath();
    ink.moveTo(left, y);
    ink.lineTo(right, y);
    ink.stroke();
    ink.fillStyle = '#38435d';
    ink.fillText((i * maximum / 4).toFixed(1), left - 8, y + 6);
  }
  ink.textAlign = 'center';
  ink.fillText('Round', (left + right) / 2, height - 5);
  ink.fillText('0', left, bottom + 23);
  ink.fillText(String(rounds), right - 12, bottom + 23);
  lines.forEach(({curve}, index) => {
    ink.strokeStyle = colors[index];
    ink.lineWidth = 3;
    ink.setLineDash([[], [9, 4], [2, 4], [12, 4, 2, 4]][index]);
    ink.beginPath();
    curve.forEach((value, i) => ink.lineTo(left + (right - left) * i / rounds,
      bottom - (bottom - top) * value / maximum));
    ink.stroke();
  });
  ink.setLineDash([]);
}

function banditDisplay() {
  const rounds = readNumber('rounds');
  const lines = runBandit(rounds, readNumber('bandit-seed'), readNumber('epsilon'), readNumber('ucb-c'));
  drawBandits(lines, rounds);
  $('bandit-results').replaceChildren(...lines.map((line, index) => {
    const row = document.createElement('tr');
    const name = document.createElement('th');
    name.scope = 'row';
    name.style.color = colors[index];
    name.textContent = line.name + ' (' + lineStyles[index] + ')';
    row.append(name);
    [line.reward, line.counts.join(' / '), line.curve.at(-1).toFixed(2)].forEach(value => {
      const cell = row.insertCell();
      cell.textContent = value;
    });
    return row;
  }));
  $('bandit-status').textContent = `Compared four policies over ${rounds} pulls per policy.`;
}

function drawGrid(id, values, activeState) {
  const {route, result} = rollout(values);
  const board = $(id);
  board.replaceChildren(...Array.from({length: 16}, (_, state) => {
    const cell = document.createElement('div');
    const terminal = state === GOAL || HAZARDS.has(state);
    const index = bestIndex(values(state));
    const direction = ['up', 'right', 'down', 'left'][index];
    const label = state === GOAL ? 'Goal' : HAZARDS.has(state) ? 'Trap' : state === START ? 'Start' : '';
    cell.className = 'cell';
    if (route.includes(state)) cell.classList.add('route');
    if (terminal) cell.classList.add(state === GOAL ? 'goal' : 'trap');
    if (activeState === state) cell.classList.add('active');
    cell.textContent = terminal ? label : `${label ? label + ' ' : ''}${LABELS[index]}`;
    cell.setAttribute('aria-label', `State ${state}: ${label || 'open'}${terminal ? '' : ', move ' + direction}${activeState === state ? ', agent is here' : ''}`);
    return cell;
  }));
  return `${result} in ${route.length - 1} moves`;
}

function showTabular(learner) {
  const route = drawGrid('q-grid', learner.values, learner.state);
  $('q-status').textContent = `${learner.episodes} episodes · Greedy evaluation: ${route}.`;
  const last = learner.last;
  $('q-update').textContent = last ? `State ${last.state} ${LABELS[last.action]} ${last.next} · reward ${last.reward} · TD target ${last.target.toFixed(3)} · Q: ${last.before.toFixed(3)} → ${last.after.toFixed(3)}` :
    'Advance one step to see the first update.';
}

function showDeep(learner) {
  const route = drawGrid('dqn-grid', learner.values);
  const loss = learner.meanLoss === null ? '—' : learner.meanLoss.toFixed(4);
  $('dqn-status').textContent = `${learner.episodes} episodes · Replay: ${learner.replaySize}/512 · Updates: ${learner.updates} · Last batch loss: ${loss} · Greedy evaluation: ${route}.`;
}

if (typeof document !== 'undefined') {
  $('bandit-form').addEventListener('submit', event => {
    event.preventDefault();
    banditDisplay();
  });
  banditDisplay();

  let q = tabular(42);
  showTabular(q);
  $('q-step').addEventListener('click', () => { q.step(); showTabular(q); });
  $('q-train').addEventListener('click', () => { q.train(75); showTabular(q); });
  $('q-reset').addEventListener('click', () => { q = tabular(42); showTabular(q); });

  let dqn = deepQ(42);
  showDeep(dqn);
  $('dqn-reset').addEventListener('click', () => { dqn = deepQ(42); showDeep(dqn); });
  $('dqn-train').addEventListener('click', async event => {
    const button = event.currentTarget;
    button.disabled = true;
    $('dqn-reset').disabled = true;
    $('dqn-status').textContent = 'Training on 100 new episodes…';
    try {
      for (let batch = 0; batch < 20; batch++) {
        await new Promise(resolve => requestAnimationFrame(resolve));
        dqn.train(5);
      }
    } finally {
      showDeep(dqn);
      button.disabled = false;
      $('dqn-reset').disabled = false;
    }
  });
}
