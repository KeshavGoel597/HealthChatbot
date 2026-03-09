'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Send, User, Bot, Loader2, Settings, PanelLeftClose, PanelLeftOpen, Languages, Volume2, FileText, ChevronDown, ChevronUp, AlertTriangle, Layers } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import { ModeToggle } from "@/components/mode-toggle"
import { ChatSidebar } from "@/components/chat/chat-sidebar"
import { ConsentModal } from "@/components/chat/consent-modal"

const BACKEND_URL = 'http://localhost:8013'

interface Message {
  role: 'user' | 'assistant'
  content: string
  audio_content?: string
  emr_fields_used?: string[]  // GDPR Art. 15 — evidence transparency
  was_compacted?: boolean
}

interface ChatSession {
  id: string
  title: string
  created_at: string
  expires_at?: string
  store_history_consent?: boolean
}

const LANGUAGES = [
  { code: "en-IN", name: "English" },
  { code: "hi-IN", name: "Hindi" },
  { code: "ta-IN", name: "Tamil" },
  { code: "te-IN", name: "Telugu" },
  { code: "ml-IN", name: "Malayalam" },
  { code: "kn-IN", name: "Kannada" },
  { code: "mr-IN", name: "Marathi" },
  { code: "gu-IN", name: "Gujarati" },
  { code: "bn-IN", name: "Bengali" },
  { code: "pa-IN", name: "Punjabi" },
  { code: "or-IN", name: "Odia" },
]

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Hello! I am Robert, your AI medical assistant. Please review the privacy notice above before we begin.' }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [model, setModel] = useState<string | null>("gemini-2.5-flash-lite")
  const [language, setLanguage] = useState<string>("en-IN")
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  // GDPR consent state
  const [showConsentModal, setShowConsentModal] = useState(true)
  const [emrConsent, setEmrConsent] = useState(false)
  const [storeHistoryConsent, setStoreHistoryConsent] = useState(false)
  const [consentGiven, setConsentGiven] = useState(false)

  // Evidence panel state (per message index)
  const [evidencePanelOpen, setEvidencePanelOpen] = useState<Record<number, boolean>>({})

  // Sidebar & Session State
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)

  // Fetch sessions on mount
  useEffect(() => {
    fetchSessions()
  }, [])

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }, [messages])

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/sessions/patient101`)
      if (res.ok) {
        const data = await res.json()
        setSessions(data)
      }
    } catch (error) {
      console.error("Failed to fetch sessions:", error)
    }
  }

  const handleConsent = (emr: boolean, storeHistory: boolean) => {
    setEmrConsent(emr)
    setStoreHistoryConsent(storeHistory)
    setConsentGiven(true)
    setShowConsentModal(false)
  }

  const handleNewChat = () => {
    setCurrentSessionId(null)
    setMessages([
      { role: 'assistant', content: 'Hello! I am Robert, your AI medical assistant. How can I help you today?' }
    ])
    setEvidencePanelOpen({})
    // Re-show consent modal for new chat (fresh consent per session — GDPR Art. 7)
    setShowConsentModal(true)
    setConsentGiven(false)
    if (window.innerWidth < 1024) setIsSidebarOpen(false)
  }

  const handleSelectSession = async (sessionId: string) => {
    setIsLoading(true)
    try {
      const res = await fetch(`${BACKEND_URL}/sessions/${sessionId}/messages`)
      if (res.ok) {
        const data = await res.json()
        // Restore consent state from the saved session
        setEmrConsent(data.emr_consent ?? false)
        setStoreHistoryConsent(data.store_history_consent ?? false)
        setConsentGiven(true)
        setShowConsentModal(false)
        setMessages(data.messages?.length > 0 ? data.messages : [
          { role: 'assistant', content: 'Hello! I am Robert. How can I help you today?' }
        ])
        setCurrentSessionId(sessionId)
        setEvidencePanelOpen({})
      }
    } catch (error) {
      console.error("Failed to load session:", error)
    } finally {
      setIsLoading(false)
      if (window.innerWidth < 1024) setIsSidebarOpen(false)
    }
  }

  const handleDeleteSession = async (sessionId: string) => {
    try {
      const res = await fetch(`${BACKEND_URL}/gdpr/sessions/${sessionId}`, { method: 'DELETE' })
      if (res.ok) {
        setSessions(prev => prev.filter(s => s.id !== sessionId))
        if (currentSessionId === sessionId) {
          handleNewChat()
        }
      }
    } catch (error) {
      console.error("Failed to delete session:", error)
    }
  }

  const handleSend = async () => {
    if (!input.trim() || !consentGiven) return

    const userMessage = { role: 'user' as const, content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      let sessionId = currentSessionId

      // If no session, create one with the current consent settings
      if (!sessionId) {
        const createRes = await fetch(`${BACKEND_URL}/sessions/patient101`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            emr_consent: emrConsent,
            store_history_consent: storeHistoryConsent,
          })
        })
        if (createRes.ok) {
          const newSession = await createRes.json()
          sessionId = newSession.id
          setCurrentSessionId(sessionId!)
          fetchSessions()
        } else {
          throw new Error("Failed to create session")
        }
      }

      // Send message (with extended timeout for TTS)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120000)

      const response = await fetch(`${BACKEND_URL}/sessions/${sessionId}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          patient_id: 'patient101',
          model: model,
          language: language,
          audio_requested: language !== "en-IN",
          emr_consent: emrConsent,                   // GDPR Art. 5(1)(a)
          store_history_consent: storeHistoryConsent, // GDPR Art. 5(1)(e)
        }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorText = await response.text()
        console.error("Server error:", response.status, errorText)
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        audio_content: data.audio_content,
        emr_fields_used: data.emr_fields_used || [],  // GDPR Art. 15
        was_compacted: data.was_compacted || false,
      }
      setMessages(prev => [...prev, assistantMessage])

      if (messages.length <= 1) {
        fetchSessions()
      }

    } catch (error) {
      console.error("Chat error:", error)
      const errorMsg = error instanceof Error ? error.message : "Unknown error"
      setMessages(prev => [...prev, {
        role: 'assistant' as const,
        content: `I'm sorry, I encountered an error: ${errorMsg}. Please ensure the backend is running.`
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const toggleEvidencePanel = (index: number) => {
    setEvidencePanelOpen(prev => ({ ...prev, [index]: !prev[index] }))
  }

  return (
    <div className="flex h-full w-full bg-background overflow-hidden">

      {/* GDPR Consent Modal — shown on first load and each new chat */}
      {showConsentModal && <ConsentModal onConsent={handleConsent} />}

      <ChatSidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
      />

      <div className="flex flex-col flex-1 h-full min-w-0 transition-all duration-300">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b shrink-0 h-[60px]">
          <div className='flex items-center gap-3'>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="mr-1 text-muted-foreground"
            >
              {isSidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}
            </Button>

            <div className="relative hidden md:block">
              <Avatar className="h-9 w-9 border">
                <AvatarImage src="/bot-avatar.png" />
                <AvatarFallback className="bg-primary/10 text-primary"><Bot size={18} /></AvatarFallback>
              </Avatar>
              <span className={`absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full border-2 border-background ${model === "gemini-2.5-flash-lite" ? 'bg-blue-500' : model === "medgemma" ? 'bg-purple-500' : 'bg-green-500'}`}></span>
            </div>

            <div>
              <h1 className="text-base font-semibold leading-none flex items-center gap-2">Robert</h1>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <p className="text-xs text-muted-foreground mt-0.5 cursor-pointer hover:text-foreground transition-colors flex items-center gap-1">
                    {model === "gemini-2.5-flash-lite" ? "Gemini 2.5 Flash" : model === "medgemma" ? "MedGemma 4B (Local)" : "Qwen 0.5B (Local)"} <Settings className="h-3 w-3" />
                  </p>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start">
                  <DropdownMenuLabel>Model Selection</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => setModel(null)} className="flex justify-between cursor-pointer">
                    Qwen 0.5B (Local) {model === null && "✓"}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setModel("medgemma")} className="flex justify-between cursor-pointer">
                    MedGemma 4B (Local) {model === "medgemma" && "✓"}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setModel("gemini-2.5-flash-lite")} className="flex justify-between cursor-pointer">
                    Gemini 2.5 Flash {model === "gemini-2.5-flash-lite" && "✓"}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            <div className="ml-1">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <Languages className="h-4 w-4 text-muted-foreground" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="max-h-[300px] overflow-y-auto">
                  <DropdownMenuLabel>Select Language</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  {LANGUAGES.map((lang) => (
                    <DropdownMenuItem key={lang.code} onClick={() => setLanguage(lang.code)} className="flex justify-between cursor-pointer">
                      {lang.name} {language === lang.code && "✓"}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ModeToggle />
          </div>
        </header>

        {/* GDPR AI Disclaimer Banner — Art. 5(1)(a) + Art. 22 */}
        <div className="flex items-center gap-2 bg-amber-500/8 border-b border-amber-500/20 px-4 py-1.5 text-xs text-amber-700 dark:text-amber-400">
          <AlertTriangle className="h-3 w-3 shrink-0" />
          <span>
            <strong>Robert is an AI assistant</strong> — not a licensed physician. Always verify medical
            information with a qualified healthcare professional before acting.
            {emrConsent
              ? <span className="ml-1 text-amber-600/80 dark:text-amber-400/60">· EMR access: <strong>on</strong></span>
              : <span className="ml-1 text-amber-600/80 dark:text-amber-400/60">· EMR access: <strong>off</strong></span>
            }
          </span>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 overflow-hidden relative">
          <ScrollArea className="h-full px-4 md:px-0" ref={scrollAreaRef}>
            <div className="max-w-3xl mx-auto py-6 space-y-6 px-2 md:px-0 pb-10">
              {messages.map((m, index) => (
                <div
                  key={index}
                  className={`flex gap-3 md:gap-4 ${m.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                >
                  <Avatar className="h-8 w-8 mt-1 shrink-0">
                    {m.role === 'user' ? (
                      <AvatarFallback className="bg-muted text-muted-foreground"><User size={16} /></AvatarFallback>
                    ) : (
                      <AvatarFallback className="bg-primary/10 text-primary"><Bot size={16} /></AvatarFallback>
                    )}
                  </Avatar>

                  <div className={`group relative max-w-[85%] sm:max-w-[75%] ${m.role === 'user' ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
                    <div className={`rounded-2xl px-4 py-2.5 text-sm shadow-sm ${m.role === 'user'
                      ? 'bg-primary text-primary-foreground rounded-tr-sm'
                      : 'bg-muted/50 text-foreground border rounded-tl-sm'
                      }`}>
                      <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>

                      {/* Audio player */}
                      {m.audio_content && (
                        <div className="mt-2 pt-2 border-t border-primary-foreground/20 flex flex-col gap-1">
                          <div className="flex items-center gap-2 text-xs opacity-70 mb-1">
                            <Volume2 size={12} />
                            <span>Audio Response</span>
                          </div>
                          <audio controls src={`data:audio/wav;base64,${m.audio_content}`} className="w-full h-8 max-w-[200px]" />
                        </div>
                      )}
                    </div>

                    {/* Metadata badges (compacted, evidence) */}
                    {m.role === 'assistant' && (
                      <div className="flex items-center gap-2 flex-wrap">
                        {/* Compaction badge */}
                        {m.was_compacted && (
                          <span className="flex items-center gap-1 text-[10px] text-muted-foreground/70 bg-muted/40 rounded-full px-2 py-0.5 border">
                            <Layers size={9} />
                            Context compacted
                          </span>
                        )}

                        {/* GDPR Art. 15 — Evidence panel toggle */}
                        {m.emr_fields_used && m.emr_fields_used.length > 0 && (
                          <button
                            onClick={() => toggleEvidencePanel(index)}
                            className="flex items-center gap-1 text-[10px] text-primary/70 hover:text-primary bg-primary/5 hover:bg-primary/10 rounded-full px-2 py-0.5 border border-primary/20 transition-colors"
                          >
                            <FileText size={9} />
                            {evidencePanelOpen[index] ? 'Hide' : 'Show'} EMR evidence
                            {evidencePanelOpen[index] ? <ChevronUp size={9} /> : <ChevronDown size={9} />}
                          </button>
                        )}
                      </div>
                    )}

                    {/* GDPR Art. 15 — Evidence panel */}
                    {m.role === 'assistant' && evidencePanelOpen[index] && m.emr_fields_used && m.emr_fields_used.length > 0 && (
                      <div className="w-full rounded-lg border border-primary/20 bg-primary/5 px-3 py-2.5 text-xs">
                        <p className="font-medium text-primary mb-1.5 flex items-center gap-1">
                          <FileText size={11} />
                          EMR Data Used in This Response
                        </p>
                        <ul className="space-y-0.5">
                          {m.emr_fields_used.map((field, i) => (
                            <li key={i} className="flex items-center gap-1.5 text-muted-foreground">
                              <span className="h-1 w-1 rounded-full bg-primary/50 shrink-0" />
                              {field}
                            </li>
                          ))}
                        </ul>
                        <p className="text-[9px] text-muted-foreground/60 mt-2">
                          GDPR Art. 15 — Right of Access: You can see exactly what data influenced this response.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex gap-4">
                  <Avatar className="h-8 w-8 mt-1 shrink-0">
                    <AvatarFallback className="bg-primary/10 text-primary"><Bot size={16} /></AvatarFallback>
                  </Avatar>
                  <div className="bg-muted/50 border rounded-2xl rounded-tl-sm px-5 py-3 flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">Thinking...</span>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Input Area */}
        <div className="p-3 md:p-4 bg-background border-t">
          <div className="max-w-3xl mx-auto relative">
            <Input
              placeholder={consentGiven ? "Message Robert..." : "Please accept the privacy notice to begin..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading || !consentGiven}
              className="pr-12 py-6 text-base rounded-full border-muted-foreground/20 focus-visible:ring-offset-0 focus-visible:ring-1 focus-visible:ring-ring bg-muted/20"
            />
            <Button
              onClick={handleSend}
              disabled={isLoading || !input.trim() || !consentGiven}
              size="icon"
              className="absolute right-1.5 top-1.5 h-9 w-9 rounded-full"
            >
              <Send className="h-4 w-4" />
              <span className="sr-only">Send</span>
            </Button>
          </div>
          <div className="mt-2 text-[10px] text-center text-muted-foreground hidden sm:block">
            AI can make mistakes. Please verify important medical information with a licensed physician.
          </div>
        </div>
      </div>
    </div>
  )
}
