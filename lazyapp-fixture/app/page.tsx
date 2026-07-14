'use client'
import { useEffect, useState } from 'react'
export default function Home() {
  const [msg, setMsg] = useState('idle')
  useEffect(() => {
    // real dynamic import — the async_target's subgraph should be DEFERRED at build
    import('./async_target').then((m) => setMsg(m.message))
  }, [])
  return (
    <main>
      <h1>dynamic import deferral test</h1>
      <p id="out">{msg}</p>
    </main>
  )
}
