// Package internal provides AI service internal implementation
// Author: Done-0
// Created: 2025-08-31
package internal

import (
	"context"
	"fmt"

	"github.com/Done-0/gin-scaffold/configs"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/prompter"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/provider"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/provider/deepseek"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/provider/gemini"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/provider/openai"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/selector"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/types"
)

// Manager is the AI manager that provides unified entry point
type Manager struct {
	config   *configs.AIConfig
	selector *selector.Selector
	prompter.Prompter
}

// New creates a new AI manager
func New(config *configs.Config) (*Manager, error) {
	return &Manager{
		config:   &config.AI,
		selector: selector.New(),
		Prompter: prompter.New(),
	}, nil
}

// Chat sends a chat completion request
func (m *Manager) Chat(ctx context.Context, req *types.ChatRequest) (*types.ChatResponse, error) {
	providerCtx, err := m.selector.Select(m.config, req.Model)
	if err != nil {
		return nil, err
	}

	var p provider.Provider
	switch providerCtx.ProviderName {
	case "deepseek":
		p, err = deepseek.New(providerCtx)
	case "gemini":
		p, err = gemini.New(providerCtx)
	case "openai":
		p, err = openai.New(providerCtx)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", providerCtx.ProviderName)
	}
	if err != nil {
		return nil, err
	}

	return p.Chat(ctx, req)
}

// ChatStream sends a streaming chat completion request
func (m *Manager) ChatStream(ctx context.Context, req *types.ChatRequest) (<-chan *types.ChatStreamResponse, error) {
	providerCtx, err := m.selector.Select(m.config, req.Model)
	if err != nil {
		return nil, err
	}

	var p provider.Provider
	switch providerCtx.ProviderName {
	case "deepseek":
		p, err = deepseek.New(providerCtx)
	case "gemini":
		p, err = gemini.New(providerCtx)
	case "openai":
		p, err = openai.New(providerCtx)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", providerCtx.ProviderName)
	}
	if err != nil {
		return nil, err
	}

	return p.ChatStream(ctx, req)
}

// ResetSelector resets the selector counters
func (m *Manager) ResetSelector() {
	m.selector.Reset()
}

// UpdateConfig updates the AI configuration
func (m *Manager) UpdateConfig(config *configs.AIConfig) {
	m.config = config
}
