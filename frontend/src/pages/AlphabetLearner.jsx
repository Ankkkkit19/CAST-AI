import { useState, useCallback, useRef, useEffect } from 'react'
import { useA11y } from '../context/AccessibilityContext'
import { useAuth } from '../context/AuthContext'
import { saveProgress } from '../services/progressService'
import './AlphabetLearner.css'

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')

// 3 ways to explain each letter with visual components
const LETTER_EXPLANATIONS = {
  A: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✋ → ✊</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a TIGHT FIST with all fingers closed</li>
          <li className="alphabet__step">✍️ Thumb goes on the SIDE (not hidden, sticking out)</li>
          <li className="alphabet__step">✍️ All fingers pressed together - no gaps!</li>
          <li className="alphabet__step">✍️ Palm can face down or forward - your choice</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Hand in FIST with thumb on side facing forward</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">✊</div>
        <p className="alphabet__story-text">
          "A" is one of the SIMPLEST signs - just a closed fist with your thumb sticking out! Think of it like you're holding something tightly in your hand. The thumb on the side is what makes it "A" and not just any fist.
          <br /><br />
          <strong>Key Point:</strong> THUMB MUST BE VISIBLE on the side - that's what makes it an "A"!
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Close all 5 fingers into a fist - tight!</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Make sure thumb is visible on the side</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Hold steady and say "A"</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Try both palm-down and palm-forward!</div>
          </div>
        </div>
      </div>
    ),
  },
  B: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">🖐️ → 🖐️</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Spread all 4 fingers WIDE APART</li>
          <li className="alphabet__step">✍️ Keep fingers straight - no bending</li>
          <li className="alphabet__step">✍️ Thumb tucks flat against palm (hidden)</li>
          <li className="alphabet__step">✍️ Palm faces toward you - hold steady</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Open hand with 4 fingers spread wide, thumb tucked</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">✋</div>
        <p className="alphabet__story-text">
          "B" is like showing someone STOP! 🛑 Your hand is FLAT and OPEN. The four fingers are spread wide apart like a fan, and your thumb is tucked in and hiding on the side.
          <br /><br />
          <strong>Key Memory:</strong> Four strong fingers spreading like a PEACE sign, but without the V!
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Spread your 4 fingers as WIDE as possible</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Keep all fingers STRAIGHT - no bending!</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Tuck thumb flat against your palm</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Try rotating your hand: palm to you, palm away</div>
          </div>
        </div>
      </div>
    ),
  },
  C: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">🖐️ → ☝️</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a FIST with all fingers together</li>
          <li className="alphabet__step">✍️ Keep thumb tucked on the side</li>
          <li className="alphabet__step">✍️ One finger (index) points UP straight</li>
          <li className="alphabet__step">✍️ Other 3 fingers stay curved and closed</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Fist with ONE index finger pointing straight up</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">☝️</div>
        <p className="alphabet__story-text">
          "C" is like pointing at the SKY! ☁️ You make a fist and stick just ONE finger up - your index finger. That one lonely finger pointing up is what makes it "C"!
          <br /><br />
          <strong>Memory Trick:</strong> Like saying "I have ONE question!" 🙋
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a tight fist with all fingers closed</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Stick ONLY your index finger straight UP</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Keep other 3 fingers curved inside fist</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Hold steady and tall - like a flagpole!</div>
          </div>
        </div>
      </div>
    ),
  },
  D: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✊ → ☝️</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a FIST with all fingers</li>
          <li className="alphabet__step">✍️ Stick your index finger UP straight</li>
          <li className="alphabet__step">✍️ Keep other fingers curled down</li>
          <li className="alphabet__step">✍️ Thumb rests on the SIDE of fist</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Fist with index up + thumb to side</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🏠</div>
        <p className="alphabet__story-text">
          "D" is like pointing UP at the roof of a house! One finger goes straight up (the roof line) and your thumb holds the corner.
          <br /><br />
          <strong>Remember:</strong> One finger up = "D" for "Door"! 🚪
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a tight fist - all fingers in</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Pop your index finger straight UP</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Make sure thumb is visible on side</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Hold strong and say "D"!</div>
          </div>
        </div>
      </div>
    ),
  },
  E: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">🖐️ → ✋</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Open ALL 5 fingers WIDE</li>
          <li className="alphabet__step">✍️ Spread them as far apart as possible</li>
          <li className="alphabet__step">✍️ Keep all fingers STRAIGHT (no bending)</li>
          <li className="alphabet__step">✍️ Palm can face any direction</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Flat hand with all 5 fingers spread wide</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🌟</div>
        <p className="alphabet__story-text">
          "E" has 3 bumps on the right side of the letter! Your 5 spread fingers look like those bumps standing tall.
          <br /><br />
          <strong>Remember:</strong> Spread 'em wide like you're catching stars! ⭐
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Open your hand super wide</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Spread ALL fingers - don't bend them</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "EEE" while showing all fingers</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Close and open 5 times fast!</div>
          </div>
        </div>
      </div>
    ),
  },
  F: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">👌 → �</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Pinch thumb and index together</li>
          <li className="alphabet__step">✍️ Make a circle shape (O shape) with them</li>
          <li className="alphabet__step">✍️ Hold middle, ring, pinky UP straight</li>
          <li className="alphabet__step">✍️ Keep the 3 fingers close together</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Circle (thumb+index) + 3 fingers up</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🍴</div>
        <p className="alphabet__story-text">
          "F" looks like a fork! The circle is the handle, and the 3 fingers up are the prongs of the fork.
          <br /><br />
          <strong>Remember:</strong> It's a tiny fork with 3 prongs! �️
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a circle with thumb and index</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Hold 3 fingers straight UP</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "FFF" with your fork hand</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Wiggle the 3 prongs! �</div>
          </div>
        </div>
      </div>
    ),
  },
  G: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">👌 → 🫱</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a C shape with thumb and index</li>
          <li className="alphabet__step">✍️ Other 3 fingers curl DOWN inside C</li>
          <li className="alphabet__step">✍️ Point your C to the SIDE</li>
          <li className="alphabet__step">✍️ It looks like a G from that angle</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: C shape pointed sideways</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🔫</div>
        <p className="alphabet__story-text">
          "G" is like the letter "C" turned sideways! It looks a little like a finger gun pointing to the side.
          <br /><br />
          <strong>Remember:</strong> It's a C that said "turn sideways!" 🔄
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a C with thumb and index</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Curl your 3 fingers inside the C</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Point the C to your side</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Say "GGG" with your G gun! 🔫</div>
          </div>
        </div>
      </div>
    ),
  },
  H: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✌️ → 🤐</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a peace sign (V shape)</li>
          <li className="alphabet__step">✍️ Keep index and middle fingers WIDE apart</li>
          <li className="alphabet__step">✍️ Both fingers point straight UP</li>
          <li className="alphabet__step">✍️ Tuck your other 3 fingers down</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Two fingers spread wide pointing up</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">⛸️</div>
        <p className="alphabet__story-text">
          "H" has two vertical lines connected at the middle! Your two spread fingers are those lines standing strong like goal posts.
          <br /><br />
          <strong>Remember:</strong> Like a goal post in football! 🏈
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a peace sign (V shape)</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Spread the 2 fingers WIDE apart</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Keep them pointing straight up</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Say "HHH" with your H! 🎉</div>
          </div>
        </div>
      </div>
    ),
  },
  I: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">☝️ → ☝️</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Hold up ONLY your pinky finger</li>
          <li className="alphabet__step">✍️ Keep it straight and TALL</li>
          <li className="alphabet__step">✍️ All other fingers curl down tight</li>
          <li className="alphabet__step">✍️ Point it straight to the sky</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Only pinky finger up, straight</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">📍</div>
        <p className="alphabet__story-text">
          "I" is the SIMPLEST letter - just one straight line! Your pinky finger is perfect for this - it's like pointing to something very important.
          <br /><br />
          <strong>Remember:</strong> One lonely finger pointing up! 🌙
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Hold your pinky straight UP</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Keep it pointing high like a tower</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "III" with your pinky point</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Wiggle just the pinky! Easy! ✨</div>
          </div>
        </div>
      </div>
    ),
  },
  J: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">☝️ → 🪝</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Hold your pinky finger UP straight</li>
          <li className="alphabet__step">✍️ Then CURVE it down smoothly</li>
          <li className="alphabet__step">✍️ Make a hook shape like a J</li>
          <li className="alphabet__step">✍️ Keep the motion graceful</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Pinky curves down into hook shape</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🪝</div>
        <p className="alphabet__story-text">
          "J" is like the letter "I" but with a hook at the bottom, just like a fishing hook! Your pinky can make this beautiful curve.
          <br /><br />
          <strong>Remember:</strong> A fishing hook that caught something! 🎣
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Point your pinky straight UP</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Curve it down smoothly to the side</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Make a hook shape - slow and pretty</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Say "JJJ" while hooking! 🪝</div>
          </div>
        </div>
      </div>
    ),
  },
  K: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✌️ → 🤐</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a peace sign (V shape)</li>
          <li className="alphabet__step">✍️ Spread index and middle WIDE</li>
          <li className="alphabet__step">✍️ Place your thumb UP between them</li>
          <li className="alphabet__step">✍️ Thumb touches from underneath</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Peace sign with thumb up between</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">⚔️</div>
        <p className="alphabet__story-text">
          "K" has one straight line and two diagonal lines meeting it. Your two fingers spread wide and thumb up in the middle look like a sword crossing!
          <br /><br />
          <strong>Remember:</strong> A sword with a cross guard! ⚔️
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a wide peace sign (V)</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Slide your thumb UP in the middle</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "KKK" with your sword hand</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Move it around - it's a K!</div>
          </div>
        </div>
      </div>
    ),
  },
  L: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">☝️👈 → �</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Point your index finger UP straight</li>
          <li className="alphabet__step">✍️ Point your thumb OUT to the side</li>
          <li className="alphabet__step">✍️ Make a 90-degree angle (like a corner)</li>
          <li className="alphabet__step">✍️ Curl your other 3 fingers down</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Index up + thumb out = L shape</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">📐</div>
        <p className="alphabet__story-text">
          "L" has one straight line and one line going across making a right angle! Your index finger up and thumb out make that perfect corner shape.
          <br /><br />
          <strong>Remember:</strong> An L-shaped corner or bracket! 📐
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Point index finger UP to the sky</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Point thumb to your SIDE</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Make a 90-degree angle corner</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Say "LLL" with your L! 📐</div>
          </div>
        </div>
      </div>
    ),
  },
  M: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">🖐️ → ☝️☝️☝️☝️</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Hold all 4 fingers UP together</li>
          <li className="alphabet__step">✍️ Keep them side by side - close</li>
          <li className="alphabet__step">✍️ All fingers STRAIGHT and tall</li>
          <li className="alphabet__step">✍️ Thumb tucks DOWN flat on palm</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: 4 fingers up + thumb hidden</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">⛰️</div>
        <p className="alphabet__story-text">
          "M" has 3 peaks in the middle like a mountain range! When you hold up your 4 fingers together, they look like those three peaks standing tall.
          <br /><br />
          <strong>Remember:</strong> Mountains standing in a line! 🏔️
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Hold all 4 fingers UP and tight</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Keep them pressed TOGETHER</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "MMM" with your mountain</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Move it up and down! 🏔️</div>
          </div>
        </div>
      </div>
    ),
  },
  N: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✌️ → ➰</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a peace sign (V shape)</li>
          <li className="alphabet__step">✍️ Keep index and middle fingers STRAIGHT</li>
          <li className="alphabet__step">✍️ Cross them at a slight angle</li>
          <li className="alphabet__step">✍️ Like an X shape - touching at middle</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Two fingers crossed at angle</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">⚡</div>
        <p className="alphabet__story-text">
          "N" has a zigzag line going through it! When you cross your two fingers, they make that cool zigzag lightning bolt pattern.
          <br /><br />
          <strong>Remember:</strong> A zigzag like lightning! ⚡
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a peace sign (V)</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Cross your 2 fingers at the middle</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Make a zigzag like lightning</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Say "NNN" with your N! ⚡</div>
          </div>
        </div>
      </div>
    ),
  },
  O: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">�️ → 🫲</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Bring ALL fingers and thumb together</li>
          <li className="alphabet__step">✍️ Make them touch - form a circle</li>
          <li className="alphabet__step">✍️ Keep the circle SMOOTH and round</li>
          <li className="alphabet__step">✍️ It should look like a O!</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: All fingers + thumb making circle</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🔴</div>
        <p className="alphabet__story-text">
          "O" is a perfect CIRCLE! When you bring all your fingers and thumb together, you make the most perfect round circle shape ever.
          <br /><br />
          <strong>Remember:</strong> A round pizza or ball! 🍕⚽
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Bring all fingers to the center</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Make them touch in a circle</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "OOO" while holding your O</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Keep it smooth and round! 🔴</div>
          </div>
        </div>
      </div>
    ),
  },
  P: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✌️ → 👇</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a peace sign (V with 2 fingers)</li>
          <li className="alphabet__step">✍️ Point your hand DOWN (flip it)</li>
          <li className="alphabet__step">✍️ Fingers point toward the ground</li>
          <li className="alphabet__step">✍️ It looks like the letter P</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Peace sign flipped upside down</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">☮️</div>
        <p className="alphabet__story-text">
          "P" is like an UPSIDE-DOWN peace sign! Instead of pointing up in the air, your fingers point down to the ground.
          <br /><br />
          <strong>Remember:</strong> A peace sign that fell down! 😄
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a peace sign</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Flip it upside down!</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Fingers point to the ground</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Say "PPP" with your P! ☮️</div>
          </div>
        </div>
      </div>
    ),
  },
  Q: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">� → 🪬</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a circle with thumb and index</li>
          <li className="alphabet__step">✍️ Make it touching - round and tight</li>
          <li className="alphabet__step">✍️ Point your PINKY down below circle</li>
          <li className="alphabet__step">✍️ The pinky is the tail of Q</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Circle + pinky down = tail</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🐭</div>
        <p className="alphabet__story-text">
          "Q" is like the letter "O" but with a TAIL! Your pinky finger is that little tail sticking down from the circle like a mouse!
          <br /><br />
          <strong>Remember:</strong> A mouse with its tail up! 🐭
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a circle with thumb+index</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Point your pinky DOWN like a tail</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "QQQ" with your Q</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Wiggle the tail! 🐭</div>
          </div>
        </div>
      </div>
    ),
  },
  R: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✌️ → ❌</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a peace sign (V)</li>
          <li className="alphabet__step">✍️ Cross your index and middle fingers</li>
          <li className="alphabet__step">✍️ Cross at the middle point</li>
          <li className="alphabet__step">✍️ They should touch where they cross</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Two fingers crossed forming X</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🚀</div>
        <p className="alphabet__story-text">
          "R" has two parts that cross right in the middle! Your crossed fingers look exactly like that perfect cross pattern.
          <br /><br />
          <strong>Remember:</strong> Like a rocket ship blasting off! 🚀
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a peace sign</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Cross your 2 fingers in the middle</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Make an X shape with them</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Say "RRR" with your R! 🚀</div>
          </div>
        </div>
      </div>
    ),
  },
  S: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✊ → 🌀</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a tight FIST with all fingers</li>
          <li className="alphabet__step">✍️ Rotate your fist - not a big movement</li>
          <li className="alphabet__step">✍️ Make a small S-shaped motion in air</li>
          <li className="alphabet__step">✍️ Wavy like a snake - sssss</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Fist making S motion in air</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🐍</div>
        <p className="alphabet__story-text">
          "S" is wavy and curvy, just like a SNAKE slithering through the grass! When you make an S motion, you look like a snake moving.
          <br /><br />
          <strong>Remember:</strong> A snake saying "sssss"! 🐍
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a tight fist - squeeze it!</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Draw an S shape in the air</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Make it smooth and wavy</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Say "SSS" while snaking! 🐍</div>
          </div>
        </div>
      </div>
    ),
  },
  T: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✊ → �</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Make a FIST with all fingers closed</li>
          <li className="alphabet__step">✍️ Stick your THUMB UP between fingers</li>
          <li className="alphabet__step">✍️ Thumb points straight up</li>
          <li className="alphabet__step">✍️ It makes the T shape!</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Fist with thumb up in middle</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">✝️</div>
        <p className="alphabet__story-text">
          "T" has one straight vertical line and one horizontal line on top like a CROSS or T-shirt! Your thumb up is that perfect T shape.
          <br /><br />
          <strong>Remember:</strong> A T-shirt or a cross! 👕✝️
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Make a tight fist</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Stick thumb UP through your fingers</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "TTT" with your T</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Tap your thumb - strong T! 👍</div>
          </div>
        </div>
      </div>
    ),
  },
  U: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✌️ → 🤦</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Hold up index and middle finger</li>
          <li className="alphabet__step">✍️ Keep them CLOSE TOGETHER (touching)</li>
          <li className="alphabet__step">✍️ Point them straight UP to sky</li>
          <li className="alphabet__step">✍️ They make the U shape!</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: 2 fingers close together pointing up</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🍽️</div>
        <p className="alphabet__story-text">
          "U" is shaped like a bowl or CUP you can drink from! When you hold your two fingers close, they make that perfect bowl shape.
          <br /><br />
          <strong>Remember:</strong> A cup for your drink! 🥤
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Hold your 2 fingers TOGETHER</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Point them straight UP high</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "UUU" with your U</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Move it around - U stays U!</div>
          </div>
        </div>
      </div>
    ),
  },
  V: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✌️ → ✌️</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Hold up index and middle finger</li>
          <li className="alphabet__step">✍️ SPREAD them WIDE apart</li>
          <li className="alphabet__step">✍️ Make a big V shape with them</li>
          <li className="alphabet__step">✍️ Keep fingers straight and tall</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: 2 fingers spread wide = V shape</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">✌️</div>
        <p className="alphabet__story-text">
          "V" is shaped like two lines meeting at a point! Your two spread fingers make that perfect V for victory and peace!
          <br /><br />
          <strong>Remember:</strong> Victory and peace! ✌️
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Hold up your 2 fingers</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">SPREAD them WIDE apart</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "VVV" while showing V</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Wiggle your V - victory! ✌️</div>
          </div>
        </div>
      </div>
    ),
  },
  W: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">🖐️ → 👋</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Hold up 4 fingers (not thumb)</li>
          <li className="alphabet__step">✍️ Keep them WIDE APART (spread)</li>
          <li className="alphabet__step">✍️ All fingers pointed UP straight</li>
          <li className="alphabet__step">✍️ They look like mountain peaks!</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: 4 fingers spread wide = W peaks</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">⛰️</div>
        <p className="alphabet__story-text">
          "W" has four peaks going up and down like a mountain range! When you spread your 4 fingers, they look like mountains standing in a row.
          <br /><br />
          <strong>Remember:</strong> Mountains in a line! 🏔️
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Hold all 4 fingers UP</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">SPREAD them really wide</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "WWW" with your mountains</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Wave your hand - W! 🏔️</div>
          </div>
        </div>
      </div>
    ),
  },
  X: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">✌️ → ❌</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Hold up index and middle finger</li>
          <li className="alphabet__step">✍️ CROSS them tightly over each other</li>
          <li className="alphabet__step">✍️ Make an X shape in the air</li>
          <li className="alphabet__step">✍️ They should touch where crossing</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: 2 fingers crossed = X shape</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">❌</div>
        <p className="alphabet__story-text">
          "X" is made of two lines that cross right in the MIDDLE! When you cross your two fingers, you make the perfect X.
          <br /><br />
          <strong>Remember:</strong> Like two swords crossing! ⚔️
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Hold your 2 fingers up</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">CROSS them in the middle</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "XXX" with your X</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Wiggle while keeping crossed! ❌</div>
          </div>
        </div>
      </div>
    ),
  },
  Y: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">👍 → 🤘</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Hold your THUMB UP straight</li>
          <li className="alphabet__step">✍️ Hold your PINKY UP straight too</li>
          <li className="alphabet__step">✍️ Tuck your 3 middle fingers DOWN</li>
          <li className="alphabet__step">✍️ It makes the Y shape!</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Thumb + pinky up = Y shape</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">🎤</div>
        <p className="alphabet__story-text">
          "Y" has two lines going up and one going down like a fork! Your thumb and pinky up with your hand down makes that perfect Y.
          <br /><br />
          <strong>Remember:</strong> Rock and roll sign! 🤘
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Hold thumb and pinky UP high</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Tuck 3 middle fingers in TIGHT</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "YYY" with your Y</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Move it around - rock on! 🤘</div>
          </div>
        </div>
      </div>
    ),
  },
  Z: {
    visual: (
      <div className="alphabet__visual-card">
        <div className="alphabet__visual-demo">☝️ → ⚡</div>
        <ol className="alphabet__steps">
          <li className="alphabet__step">✍️ Point your index finger UP</li>
          <li className="alphabet__step">✍️ Draw a Z shape in the air</li>
          <li className="alphabet__step">✍️ Go diagonal down, then across, then diagonal</li>
          <li className="alphabet__step">✍️ Make it BIG and clear!</li>
        </ol>
        <div className="alphabet__tip">💡 Precise ISL: Index finger drawing Z in air</div>
      </div>
    ),
    story: (
      <div className="alphabet__story-card">
        <div className="alphabet__story-emoji">⚡</div>
        <p className="alphabet__story-text">
          "Z" is a ZIGZAG like lightning! When you draw a Z shape with your finger, you're making the letter Z just like on paper.
          <br /><br />
          <strong>Remember:</strong> Zigzag like lightning! ⚡
        </p>
      </div>
    ),
    practice: (
      <div className="alphabet__practice-card">
        <div className="alphabet__practice-steps">
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">1</div>
            <div className="alphabet__step-text">Point your index finger UP</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">2</div>
            <div className="alphabet__step-text">Draw BIG Z letters in the air</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">3</div>
            <div className="alphabet__step-text">Say "ZZZ" while drawing</div>
          </div>
          <div className="alphabet__practice-step">
            <div className="alphabet__step-num">4</div>
            <div className="alphabet__step-text">Keep zigzagging - Z it up! ⚡</div>
          </div>
        </div>
      </div>
    ),
  },
}

export default function AlphabetLearner({ setPage }) {
  const { settings } = useA11y()
  const { currentUser } = useAuth()
  const [stepIndex, setStepIndex] = useState(0)
  const [activeExplanation, setActiveExplanation] = useState(0)
  const [animatingIn, setAnimatingIn] = useState(true)
  const [completedLetters, setCompletedLetters] = useState(new Set())
  const selectedLetter = ALPHABET[stepIndex]

  // Save progress to Firebase whenever a new letter is visited
  useEffect(() => {
    if (currentUser) {
      const newCompleted = new Set(completedLetters);
      newCompleted.add(selectedLetter);
      setCompletedLetters(newCompleted);
      const pct = Math.round((newCompleted.size / 26) * 100);
      saveProgress(currentUser.uid, 'alphabets', {
        completed: [...newCompleted],
        currentLetter: selectedLetter,
        currentStep: stepIndex + 1,
        total: 26,
        percent: pct,
        email: currentUser.email,
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stepIndex]);


  const speak = useCallback((text) => {
    if (!settings.tts || !window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    
    setTimeout(() => {
      const utt = new SpeechSynthesisUtterance(text);
      utt.rate = 0.9;
      utt.pitch = 1.2;
      
      window._verbalUtterances = window._verbalUtterances || [];
      window._verbalUtterances.push(utt);
      
      utt.onend = () => {
         window._verbalUtterances = window._verbalUtterances.filter(u => u !== utt);
      };
      
      window.speechSynthesis.speak(utt);
    }, 50);
  }, [settings.tts])

  const goToStep = (idx) => {
    const next = Math.max(0, Math.min(25, idx))
    setAnimatingIn(false)
    setTimeout(() => {
      setStepIndex(next)
      setActiveExplanation(0)
      setAnimatingIn(true)
      speak(ALPHABET[next])
    }, 200)
  }

  const handleLetterSelect = (letter) => {
    const idx = ALPHABET.indexOf(letter)
    if (idx >= 0) goToStep(idx)
  }

  const nextExplanation = () => {
    setAnimatingIn(false)
    setTimeout(() => {
      setActiveExplanation((a) => (a + 1) % 3)
      setAnimatingIn(true)
    }, 200)
  }

  const explanations = LETTER_EXPLANATIONS[selectedLetter] || {}
  const explanationTypes = [
    { key: 'visual', icon: '👁️', label: 'How it looks' },
    { key: 'story', icon: '🎭', label: 'Remember it' },
    { key: 'practice', icon: '💪', label: 'Practice it' },
  ]
  const currentExpl = explanationTypes[activeExplanation]

  return (
    <main id="main-content" className="alphabet-learner">
      <div className="container">
        {/* Header */}
        <div className="alphabet__header fade-in-up">
          <div>
            <h1 className="alphabet__title">🔤 Learn the Alphabet!</h1>
            <p className="alphabet__subtitle">Step-by-step A–Z. Use Prev/Next or tap any letter.</p>
          </div>
          <button className="btn btn-ghost" onClick={() => setPage('dashboard')}>
            ← Back to Courses
          </button>
        </div>

        {/* Step navigation */}
        <div className="alphabet__step-nav glass-card fade-in-up">
          <span className="alphabet__step-label">Step {stepIndex + 1} of 26</span>
          <div className="alphabet__step-buttons">
            <button className="btn btn-ghost btn-sm" onClick={() => goToStep(stepIndex - 1)} disabled={stepIndex === 0}>
              ← Previous
            </button>
            <button className="btn btn-primary btn-sm" onClick={() => goToStep(stepIndex + 1)} disabled={stepIndex === 25}>
              Next →
            </button>
          </div>
        </div>

        {/* Main Learning Section */}
        <div className="alphabet__main">
          {/* Large Letter Display */}
          <div className="alphabet__display glass-card fade-in-up delay-1">
            <div className={`alphabet__letter-big ${animatingIn ? 'animate-in' : ''}`}>
              {selectedLetter}
            </div>
            <p className="alphabet__letter-spoken" onClick={() => speak(selectedLetter)} role="button" tabIndex="0">
              🔊 Click to hear: "{selectedLetter}"
            </p>
          </div>

          {/* Explanation Cards */}
          <div className="alphabet__explanation glass-card fade-in-up delay-2">
            <div className="alphabet__exp-header">
              <span className="alphabet__exp-icon">{currentExpl.icon}</span>
              <span className="alphabet__exp-label">{currentExpl.label}</span>
            </div>
            <div className={`alphabet__exp-content ${animatingIn ? 'animate-in' : ''}`}>
              {explanations[currentExpl.key]}
            </div>
            <div className="alphabet__exp-nav">
              <button className="btn btn-sm btn-ghost" onClick={nextExplanation}>
                → Next Way ({activeExplanation + 1}/3)
              </button>
            </div>
          </div>
        </div>

        {/* Alphabet Grid */}
        <div className="alphabet__grid-section fade-in-up delay-3">
          <p className="alphabet__grid-label">Step {stepIndex + 1} of 26 — or tap any letter:</p>
          <div className="alphabet__grid">
            {ALPHABET.map((letter, i) => (
              <button
                key={letter}
                className={`alphabet__cell ${selectedLetter === letter ? 'active' : ''} ${i < stepIndex ? 'done' : ''}`}
                onClick={() => handleLetterSelect(letter)}
                aria-label={`Letter ${letter}`}
                aria-pressed={selectedLetter === letter}
              >
                {letter}
              </button>
            ))}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="alphabet__progress fade-in-up delay-4">
          <div className="progress-bar">
            <div
              className="progress-bar__fill"
              style={{ width: `${((stepIndex + 1) / 26) * 100}%` }}
            />
          </div>
          <p className="alphabet__progress-text">
            Step {stepIndex + 1} of 26 letters
          </p>
        </div>
      </div>
    </main>
  )
}
